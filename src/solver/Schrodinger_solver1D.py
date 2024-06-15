import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from typing import Optional, Callable, Union, Tuple, List

""" constant """
h = 6.62607015e-34      # Planck constant (J s)
h_bar = h / (2 * np.pi) # reduced Planck constant (J s)
m0 = 9.10938356e-31     # electron mass (kg)
COMPLEX_DTYPE = np.complex128

""" operators """
def construct_Laplacian_operator(size:int, dx:float = 1.0, dtype:np.dtype = COMPLEX_DTYPE) -> sp.spmatrix:
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

def construct_kinetic_operator(size:int, dx:float = 1.0, h_bar:float = h_bar, m:float = m0, dtype:np.dtype = COMPLEX_DTYPE) -> sp.spmatrix:
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

def construct_potential_operator(V:np.ndarray, dtype:np.dtype = COMPLEX_DTYPE) -> sp.spmatrix:
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

def construct_hamiltonian_operator(V:np.ndarray, dx:float = 1.0, h_bar:float = h_bar, m:float = m0, dtype:np.dtype = COMPLEX_DTYPE) -> sp.spmatrix:
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
                      dtype:np.dtype = COMPLEX_DTYPE,
    ) -> np.ndarray:
    if method.upper() in ('LINEAR', 'L'):
        vec = np.zeros_like(x, dtype=dtype)
        vec[mask_PML_left]  = strength * np.linspace(1, 0, np.sum(mask_PML_left))
        vec[mask_PML_right] = strength * np.linspace(0, 1, np.sum(mask_PML_right))
        return sp.diags([1j * vec], [0], shape=(x.size, x.size), dtype=dtype)
    
    elif method.upper() in ('QUADRATIC', 'Q'):
        vec = np.zeros_like(x, dtype=dtype)
        vec[mask_PML_left]  = strength * np.linspace(1, 0, np.sum(mask_PML_left))**2
        vec[mask_PML_right] = strength * np.linspace(0, 1, np.sum(mask_PML_right))**2
        return sp.diags([1j * vec], [0], shape=(x.size, x.size), dtype=dtype)
    
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
    Psis = Psis.copy()
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
    
    if Psis.ndim == 1:
        Psis = Psis[None,:]
        only_one_state = True
    else:
        only_one_state = False

    Psis = Psis.copy()
    for ii in range(Psis.shape[0]):
        area = np.trapz(np.abs(Psis[ii])**2, x)
        Psis[ii] /= np.sqrt(area)

    if only_one_state:
        Psis = Psis[0]
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
    method : method to solve eigenvalue problem ['EIGSH', 'EIGS', None]

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
                         remove_PML:bool = True,
                         dtype:np.dtype = COMPLEX_DTYPE,
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
    method : method to solve eigenvalue problem ['EIGSH', 'EIGS', None]
    add_PML : add PML layers
    remove_PML : remove PML layers
    
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
    T = construct_kinetic_operator(size = V.size, dx = x[1] - x[0], h_bar = h_bar, m = m, dtype = dtype)
    V = construct_potential_operator(V, dtype = dtype)
    H = T + V
    if add_PML:
        PML_damping_strength = -(h_bar**2/2/m) * kwargs.get('PML_damping_strength', 1.0)
        PML_damping_method   = kwargs.get('PML_damping_method', 'QUADRATIC')
        H = H + construct_PML_vec(x, mask_PML_left=mask_PML_left, mask_PML_right=mask_PML_right, strength=PML_damping_strength, method=PML_damping_method)

    # solve eigenvalue problem
    E, Psis = solve_eigenvalue(H, num_eigenvalues = num_eigenvalues, method=method)
    if add_PML and remove_PML:
        mask_PML = mask_PML_left | mask_PML_right
        Psis = Psis[:,~mask_PML]
        x = x[~mask_PML]

    # normalize
    Psis = normalize_psi_by_area(x=x, Psis=Psis)
    return E.real, Psis


""" post analysis """
def cal_momentum(x:np.ndarray, 
                 Psis:np.ndarray,
                 h_bar:float = h_bar, 
                 normalize:bool = True, 
                 expectation:bool = False,
    ) -> np.ndarray:
    """
    calculate momentum
    
    usage
    -----
    momentum = cal_momentum(x, Psis, h_bar)

    Parameters
    ----------
    x : grid points
    Psis : wavefunctions
    h_bar : reduced Planck constant
    normalize : normalize wavefunctions
    expectation : calculate expectation value

    Returns
    -------
    momentum : momentum 
               if expectation is True, return expectation value
               else, return momentum array at each grid point
    
    """
    # normalize wavefunctions
    if Psis.ndim == 1:
        Psis = Psis[None,:]
        only_one_state = True
    else:
        only_one_state = False
    if normalize:
        Psis = normalize_psi_by_area(x=x, Psis=Psis)

    # calculate momentum at each grid point
    dx = x[1] - x[0]
    momentum = np.array( [ psi.conj() * -1j * h_bar * np.gradient(psi, dx) for psi in Psis ] )

    # calculate expectation value / not observable -> still complex
    if expectation:
        momentum = np.trapz(momentum, x)
    
    if only_one_state == 1:
        momentum = momentum[0]
    return momentum

def cal_kinetic_energy(x:np.ndarray, 
                       Psis:np.ndarray, 
                       h_bar:float = h_bar, 
                       m:float = m0, 
                       Psis_0:Optional[np.ndarray] = None, 
                       Psis_N:Optional[np.ndarray] = None, 
                       normalize:bool = True,
                       expectation:bool = False,
                       to_real:bool = True,
    ) -> np.ndarray:
    """
    calculate kinetic energy
    
    usage
    -----
    kinetic_energy = cal_kinetic_energy(x, Psis, h_bar, m, Psis_0 = None, Psis_N = None, normalize = True, expectation = True, to_real = True)

    Parameters
    ----------
    x : grid points
    Psis : wavefunctions
    h_bar : reduced Planck constant
    m : mass of particle
    Psis_0 : wavefunction before x[0]
    Psis_N : wavefunction after x[-1]
    normalize : normalize wavefunctions
    expectation : calculate expectation value
    to_real : return real part of kinetic energy

    Returns
    -------
    kinetic_energy : kinetic energy
                     if expectation is True, return expectation value
                     else, return kinetic energy array at each grid point`
    
    """

    if Psis.ndim == 1:
        Psis = np.array([Psis])
        only_one_state = True
    else:
        only_one_state = False
    
    # normalize wavefunctions
    if normalize:
        Psis = normalize_psi_by_area(x=x, Psis=Psis)
    
    # set boundary condition
    Psis_0 = np.zeros(Psis.shape[0], dtype=Psis.dtype) if Psis_0 is None else Psis_0
    Psis_N = np.zeros(Psis.shape[0], dtype=Psis.dtype) if Psis_N is None else Psis_N
    
    # calculate kinetic energy at each grid point
    kinetic_energy = np.zeros(Psis.shape, dtype=Psis.dtype)
    dx = x[1] - x[0]
    
    kinetic_energy[:, 1:-1] = (Psis[:, 0:-2] - 2 * Psis[:, 1:-1] + Psis[:, 2:]) / dx**2
    
    kinetic_energy[:, 0]    = (Psis_0        - 2 * Psis[:, 0]    + Psis[:, 1] ) / dx**2
    kinetic_energy[:, -1]   = (Psis[:, -2]   - 2 * Psis[:, -1]   + Psis_N     ) / dx**2
    kinetic_energy *= -(h_bar**2 / 2 / m)
    kinetic_energy = Psis.conj() * kinetic_energy

    # calculate expectation value
    if expectation:
        kinetic_energy = np.trapz(kinetic_energy, x)
    
    # convert to real value / observable
    if to_real:
        kinetic_energy = kinetic_energy.real
    
    if only_one_state:
        kinetic_energy = kinetic_energy[0]
    return kinetic_energy

def cal_potential_energy(x:np.ndarray,
                         Psis:np.ndarray,
                         V:Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
                         normalize:bool = True,
                         expectation:bool = False,
                         to_real:bool = True,
    ) -> np.ndarray:
    """
    calculate potential energy

    usage
    -----
    potential_energy = cal_potential_energy(x, V, Psis, normalize = True, expectation = True, to_real = True)

    Parameters
    ----------
    x : grid points
    V : potential energy / can be a function or an array
    Psis : wavefunctions
    normalize : normalize wavefunctions
    expectation : calculate expectation value
    to_real : return real part of potential energy

    Returns
    -------
    potential_energy : potential energy
                       if expectation is True, return expectation value
                       else, return potential energy array at each grid point
    
    """
    # normalize wavefunctions
    if Psis.ndim == 1:
        Psis = Psis[None,:]
        is_one_state = True
    else:
        is_one_state = False
    if normalize:
        Psis = normalize_psi_by_area(x=x, Psis=Psis)
    
    # get potential energy
    if callable(V):
        V = V(x)
    if V.size != x.size:
        raise ValueError(f'The size of V ({V.shape}) must have the same size as x ({x.shape})')
    
    # calculate potential energy at each grid point
    potential_energy = Psis.conj() * V[None,:] * Psis

    # calculate expectation value
    if expectation:
        potential_energy = np.trapz(potential_energy, x)
    
    # convert to real value / observable
    if to_real:
        potential_energy = potential_energy.real
    
    if is_one_state:
        potential_energy = potential_energy[0]
    return potential_energy
                        

""" plot """
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
            scale = 1.0
        elif isinstance(scaling, (float, int)):
            scale = float(scaling)
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

    ax.set_xlabel(kwargs.get('xlabel', 'x'))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    ax.set_xlim(kwargs.get('xlim', [x[0], x[-1]]))
    ax.set_ylim(kwargs.get('ylim', [None, None]))
    ax.set_title(kwargs.get('title', 'Wavefunctions and Potential Energy'))
    
    return ax

def plot_kinetic_energy_and_potential_energy_at_x(
        x:np.ndarray,
        Psis:np.ndarray,
        V:Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
        h_bar:float = h_bar, 
        m:float = m0, 
        Psis_0:Optional[np.ndarray] = None, 
        Psis_N:Optional[np.ndarray] = None, 
        normalize:bool = True,
        ax:Optional[plt.Axes] = None,
        lablels:Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Axes:
    """
    plot kinetic energy and potential energy at each grid point

    """
    if Psis.ndim == 1:
        Psis = Psis[None,:]

    # calculate kinetic energy
    T = cal_kinetic_energy(x = x, Psis = Psis, h_bar = h_bar, m = m, Psis_0 = Psis_0, Psis_N = Psis_N, normalize = normalize, expectation = False, to_real = True)

    # calculate potential energy
    V = cal_potential_energy(x = x, Psis = Psis, V = V, normalize = normalize, expectation = False, to_real = True)

    # get controls
    xlabel = kwargs.get('xlabel', 'x')
    ylabel = kwargs.get('ylabel', 'Energy')
    fontsize = kwargs.get('fontsize', 12)
    plot_kinetic_energy = kwargs.get('plot_kinetic_energy', True)
    plot_potential_energy = kwargs.get('plot_potential_energy', True)
    plot_total_energy = kwargs.get('plot_total_energy', True)

    labels = lablels if lablels is not None else [f'state-{ii}' for ii in range(Psis.shape[0])]
    if len(labels) != Psis.shape[0]:
        raise ValueError(f'The size of labels ({len(labels)}) must have the same size as Psis ({Psis.shape[0]})')

    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))
    
    # plot
    for ii in range(Psis.shape[0]):
        if plot_kinetic_energy:
            ax.plot(x, T[ii], 'r-', label=f'K.E./{labels[ii]}')
        if plot_potential_energy:
            ax.plot(x, V[ii], 'b-', label=f'V/{labels[ii]}')
        if plot_total_energy:
            ax.plot(x, T[ii] + V[ii], 'k-', label=f'Total/{labels[ii]}')
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    return ax
