import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from typing import Optional, Callable, Union, Tuple

""" constant """
h = 6.62607015e-34      # Planck constant (J s)
h_bar = h / (2 * np.pi) # reduced Planck constant (J s)
m0 = 9.10938356e-31     # electron mass (kg)

""" operators """
def construct_Laplacian_operator(size:int, dx:float = 1.0, dtype = np.complex128) -> sp.spmatrix:
    """ 
    Construct Laplacian operator (sparse matrix) in 1D
    
    Parameters
    ----------
    size : number of grid points
    dx : grid spacing
    dtype : data type of the matrix

    Returns
    -------
    T : Laplacian operator

    """
    diagonals = [
        -2 * np.ones(size) ,
        np.ones(size-1),
        np.ones(size-1) / dx**2
    ]
    offsets = [0, -1, 1]
    T = sp.diags(diagonals, offsets, shape=(size, size), dtype=dtype) / dx**2
    return T

def construct_kinetic_operator(size:int, dx:float = 1.0, h_bar:float = h_bar, m:float = m0, dtype = np.complex128) -> sp.spmatrix:
    """
    construct kinetic operator (sparse matrix) in 1D

    Parameters
    ----------
    size : number of grid points
    dx : grid spacing
    h_bar : reduced Planck constant
    m : mass of particle
    dtype : data type of the matrix

    Returns
    -------
    T : kinetic operator

    """
    return -(h_bar**2/2/m) * construct_Laplacian_operator(size = size, dx = dx, dtype = dtype)

def construct_potential_operator(V:np.ndarray, dtype = np.complex128) -> sp.spmatrix:
    """
    construct potential operator (sparse matrix) in 1D
    
    Parameters
    ----------
    V : potential energy
    dtype : data type of the matrix
    
    Returns
    -------
    V : potential operator
    
    """
    return sp.diags([V], [0], shape=(V.size, V.size), dtype=dtype)

def construct_hamiltonian_operator(V:np.ndarray, dx:float = 1.0, h_bar:float = h_bar, m:float = m0, dtype = np.complex128) -> sp.spmatrix:
    """
    construct Hamiltonian operator (sparse matrix) in 1D

    Parameters
    ----------
    V : potential energy
    dx : grid spacing
    h_bar : reduced Planck constant
    m : mass of particle
    dtype : data type of the matrix

    Returns
    -------
    H : Hamiltonian operator

    """
    T = construct_kinetic_operator(size = V.size, dx = dx, h_bar = h_bar, m = m, dtype = dtype)
    V = construct_potential_operator(V, dtype = dtype)
    return T + V

def construct_PML_vec(x:np.ndarray, 
                      mask_PML_left:np.ndarray,
                      mask_PML_right:np.ndarray,
                      strength:float = 1.0, 
                      method:str = 'QUADRATIC',
    ) -> np.ndarray:
    if method.upper() in ('LINEAR', 'L'):
        vec = np.zeros_like(x, dtype=np.complex128)
        vec[mask_PML_left]  = strength * np.linspace(1, 0, np.sum(mask_PML_left))
        vec[mask_PML_right] = strength * np.linspace(0, 1, np.sum(mask_PML_right))
        return sp.diags([1j * vec], [0], shape=(x.size, x.size), dtype=np.complex128)
    
    elif method.upper() in ('QUADRATIC', 'Q'):
        vec = np.zeros_like(x, dtype=np.complex128)
        vec[mask_PML_left]  = strength * np.linspace(1, 0, np.sum(mask_PML_left))**2
        vec[mask_PML_right] = strength * np.linspace(0, 1, np.sum(mask_PML_right))**2
        return sp.diags([1j * vec], [0], shape=(x.size, x.size), dtype=np.complex128)
    
    else:
        raise ValueError(f'Invalid method ({method})')

""" psi """
def translate_psi_to_prob(Psis:np.ndarray) -> np.ndarray:
    """
    translate wavefunction to probability density
    
    Parameters
    ----------
    Psis : wavefunctions
    
    Returns
    -------
    prob : probability density
    
    """
    return np.abs(Psis)**2

def normalize_psi_max(Psis:np.ndarray, factors:Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """
    normalize wavefunction
    
    Parameters
    ----------
    Psis : wavefunctions
    factors : normalization factors
    
    Returns
    -------
    Psis : normalized wavefunctions
    
    """
    for ii in range(Psis.shape[0]):
        prob = np.abs(Psis[ii])**2
        prob_max = np.max(prob)
        Psis[ii] /= np.sqrt(prob_max)
        if factors is not None:
            Psis[ii] *= factors[ii] if isinstance(factors, np.ndarray) else factors
    return Psis

def normalize_psi_by_area(x:np.ndarray, Psis:np.ndarray) -> np.ndarray:
    """
    normalize wavefunction by area
    
    Parameters
    ----------
    x : grid points
    Psis : wavefunctions
    
    Returns
    -------
    Psis : normalized wavefunctions
    
    """
    for ii in range(Psis.shape[0]):
        area = np.trapz(np.abs(Psis[ii])**2, x)
        Psis[ii] /= np.sqrt(area)

        factor = np.sign(Psis[ii])
        Psis[:,ii] *= factor[np.argmax(np.abs(Psis[ii]))]
    return Psis

""" Solver """
def add_PML_layers(x:np.ndarray,
                   width:Union[float,Tuple[float]] = 1.0,
                   ratio:Union[float,Tuple[float]] = 0.1,
                   method:str = 'ratio',
    ):
    # get extended width
    if method.upper() in ('RATIO', ):
        L = x[-1] - x[0]
        if isinstance(ratio, tuple):
            left_width  = ratio[0] * L
            right_width = ratio[1] * L
        else:
            left_width = right_width = ratio * L
    elif method.upper() in ('WIDTH', ):
        if isinstance(width, tuple):
            left_width = width[0]
            right_width = width[1]
        else:
            left_width = right_width = width
    else:
        raise ValueError(f'Invalid method ({method})')

    # dx
    dx = x[1] - x[0]

    # added PML
    x_left  = np.arange( x[0] - dx,  x[0] - left_width  - dx, -dx)[::-1]
    x_right = np.arange(x[-1] + dx, x[-1] + right_width + dx,  dx)
    new_x   = np.concatenate([x_left, x, x_right])

    # mask
    mask_PML_left = np.zeros_like(new_x, dtype=bool)
    mask_PML_left[:x_left.size] = True
    mask_PML_right = np.zeros_like(new_x, dtype=bool)
    mask_PML_right[-x_right.size:] = True

    return new_x, mask_PML_left, mask_PML_right

def solve_eigenvalue(H:sp.spmatrix, num_eigenvalues:Optional[int] = None, method:Optional[str] = None) -> np.ndarray:
    """
    solve eigenvalue problem

    Parameters
    ----------
    H : Hamiltonian operator
    num_eigenvalues : number of eigenvalues to solve

    Returns
    -------
    eigenvalues : eigenvalues
    eigenvectors : eigenvectors

    """
    if num_eigenvalues is None:
        num_eigenvalues = H.shape[0]

    if method is None:
        eigenvalues, eigenvectors = np.linalg.eigh(H.todense())
    elif method.upper() in ('EIGSH', 'EIGENSH', 'EIGENSHARP', 'SHARP', 'S'):
        eigenvalues, eigenvectors = spla.eigsh(H, k=num_eigenvalues, which='SM')
    elif method.upper() in ('EIGS', 'EIGEN', 'E'):
        eigenvalues, eigenvectors = spla.eigs(H, k=num_eigenvalues, which='SM')
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(H.todense())

    # to array
    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)

    # sort
    eigenvectors = eigenvectors.transpose()
    idx = eigenvalues.argsort()  # 获取排序索引
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx]

    if num_eigenvalues is not None:
        eigenvalues = eigenvalues[:num_eigenvalues]
        eigenvectors = eigenvectors[:num_eigenvalues]
    
    return eigenvalues, eigenvectors

def solve_Schrodinger_eq(x:np.ndarray, 
                         potential_fun:Callable[[np.ndarray], np.ndarray],
                         h_bar:float = h_bar, 
                         m:float = m0, 
                         num_eigenvalues:Optional[int] = None,
                         method:Optional[str] = None,
                         add_PML:Optional[bool] = False,
                         **kwargs,
    ) -> np.ndarray:
    """
    solve 1D time-independent Schrodinger equation
    
    Parameters
    ----------
    x : grid points
    potential_fun : potential energy function
    h_bar : reduced Planck constant
    m : mass of particle
    num_eigenvalues : number of eigenvalues to solve
    
    Returns
    -------
    E : energy eigenvalues
    psi : wavefunctions
    
    """
    
    # get grid values
    V = potential_fun(x)
    if add_PML:
        PML_width  = kwargs.get('PML_width', 1.0)
        PML_ratio  = kwargs.get('PML_ratio', 0.1)
        PML_expanding_method = kwargs.get('PML_expanding_method', 'ratio')
        x, mask_PML_left, mask_PML_right = add_PML_layers(x, width=PML_width, ratio=PML_ratio, method=PML_expanding_method)
        V = np.concatenate([np.ones_like(x[mask_PML_left]) * V[0], V, np.ones_like(x[mask_PML_right]) * V[-1]])

    # construct Hamiltonian operator
    T = construct_kinetic_operator(size = V.size, dx = x[1] - x[0], h_bar = h_bar, m = m, dtype = np.complex128)
    V = construct_potential_operator(V, dtype = np.complex128)
    H = T + V
    if add_PML:
        PML_damping_strength = -(h_bar**2/2/m) * kwargs.get('PML_damping_strength', 1.0)
        PML_damping_method   = kwargs.get('PML_damping_method', 'QUADRATIC')
        H = H + construct_PML_vec(x, mask_PML_left=mask_PML_left, mask_PML_right=mask_PML_right, strength=PML_damping_strength, method=PML_damping_method)

    # solve eigenvalue problem
    E, Psis = solve_eigenvalue(H, num_eigenvalues = num_eigenvalues, method=method)
    if add_PML:
        mask_PML = mask_PML_left | mask_PML_right
        Psis = Psis[:,~mask_PML]
        x = x[~mask_PML]

    # normalize
    Psis = normalize_psi_by_area(x=x, Psis=Psis)
    return E.real, Psis

def plot_wavefunctions(x:np.ndarray, 
                       Psis:np.ndarray, 
                       offsets:Optional[np.ndarray] = None, 
                       labels:Optional[np.ndarray] = None, 
                       scaling:Optional[Union[float, np.ndarray]] = None,
                       format:str = 'real', 
                       ax:Optional[plt.Axes] = None,
                       **kwargs,
    ) -> plt.Axes:
    """
    plot wavefunctions
    
    Parameters
    ----------
    x : grid points
    Psis : wavefunctions
    E : energy eigenvalues
    V : potential energy
    
    """
    
    x = np.asarray(x)
    Psis = np.asarray(Psis)

    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    # controls
    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'Psi(x)')
    fontsize = kwargs.get('fontsize', 12)
    linestyle = kwargs.get('linestyle', 'k-')

    # plot
    for ii in range(Psis.shape[0]):
        offset = 0 if offsets is None else offsets[ii]
        label = None if labels is None else labels[ii]
        
        if scaling is None:
            scale = 1
        elif isinstance(scaling, float):
            scale = scaling
        else:
            scale = scaling[ii]

        if format.upper() in ('REAL', 'RE', 'R'):
            ax.plot(x, Psis[ii].real * scale + offset, linestyle, label=label)
        elif format.upper() in ('IMAGINARY', 'IMAG', 'IM', 'I'):
            ax.plot(x, Psis[ii].imag * scale+ offset, linestyle, label=label)
        elif format.upper() in ('ABSOLUTE', 'ABS', 'A'):
            ax.plot(x, np.abs(Psis[ii]) * scale + offset, linestyle, label=label)
        elif format.upper() in ('ANGLE', 'PHASE', 'P'):
            ax.plot(x, np.angle(Psis[ii], deg=bool) * scale + offset, linestyle, label=label)
        elif format.upper() in ('PROBABILITY', 'PROB', 'P'):
            ax.plot(x, np.abs(Psis[ii])**2 * scale + offset, linestyle, label=label)
        else: 
            raise ValueError(f'Invalid format ({format})')

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    return ax
    
def plot_potential(x:np.ndarray, V:np.ndarray, ax:Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    plot potential energy
    
    Parameters
    ----------
    x : grid points
    V : potential energy
    
    """
    
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    # controls
    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'V(x)')
    fontsize = kwargs.get('fontsize', 12)
    linestyle = kwargs.get('linestyle', 'r-')

    # plot
    ax.plot(x, V, linestyle)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    return ax

def plot_energy_level(x:np.ndarray, E:np.ndarray, labels:Optional[np.ndarray] = None, ax:Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    plot energy levels
    
    Parameters
    ----------
    x : grid points
    E : energy eigenvalues
    
    """
    
    x = np.asarray(x)
    E = np.asarray(E)

    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    # controls
    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'Energy')
    fontsize = kwargs.get('fontsize', 12)
    linestyle = kwargs.get('linestyle', 'r--')
    color = kwargs.get('color', 'r')

    # plot
    for ii in range(E.size):
        ax.plot(x, np.ones(x.size) * E[ii], linestyle)

        if labels is not None:
            ax.text( x[-1], E[ii], labels[ii], fontsize=fontsize, color = color)
        else:
            ax.text( x[-1], E[ii], f'$E_{ii}$', fontsize=fontsize, color = color)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    return ax

def plot_wavefunctions_and_potential(x:np.ndarray, 
                                     E:np.ndarray,
                                     Psis:np.ndarray, 
                                     V:np.ndarray, 
                                     labels:Optional[np.ndarray] = None, 
                                     scaling:Optional[Union[float, np.ndarray]] = None,
                                     format:str = 'real', 
                                     ax:Optional[plt.Axes] = None,
                                     **kwargs,
    ) -> plt.Axes:
    """
    plot wavefunctions and potential energy
    
    Parameters
    ----------
    x : grid points
    Psis : wavefunctions
    V : potential energy
    
    """
    
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    
    # plot
    plot_potential(x, V, ax=ax, **kwargs)
    plot_wavefunctions(x, Psis, offsets=E, labels=labels, scaling=scaling, format=format, ax=ax, **kwargs)
    plot_energy_level(x, E, ax=ax, labels = labels, **kwargs)

    ax.set_ylabel('')
    ax.set_title(kwargs.get('title', 'Wavefunctions and Potential Energy'))
    return ax

