import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from typing import Optional, Callable, Union

""" constant """
h = 6.62607015e-34      # Planck constant (J s)
h_bar = h / (2 * np.pi) # reduced Planck constant (J s)
m0 = 9.10938356e-31     # electron mass (kg)

""" Laplacian """
def construct_Laplacian_operator(size:int, dx:float = 1.0, dtype = np.float64) -> sp.spmatrix:
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
        -2 * np.ones(size) / dx**2,
        np.ones(size-1) / dx**2,
        np.ones(size-1) / dx**2
    ]
    offsets = [0, -1, 1]
    T = sp.diags(diagonals, offsets, shape=(size, size), dtype=dtype)
    return T

def construct_kinetic_operator(size:int, dx:float = 1.0, h_bar:float = h_bar, m:float = m0, dtype = np.float64) -> sp.spmatrix:
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

def construct_potential_operator(V:np.ndarray, dtype = np.float64) -> sp.spmatrix:
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

def construct_hamiltonian_operator(V:np.ndarray, dx:float = 1.0, h_bar:float = h_bar, m:float = m0, dtype = np.float64) -> sp.spmatrix:
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
def solve_eigenvalue(H:sp.spmatrix, num_eigenvalues:Optional[int] = None) -> np.ndarray:
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
    eigenvalues, eigenvectors = spla.eigs(H, k=num_eigenvalues, which='SM')
    return eigenvalues, eigenvectors.transpose()

def solve_Schrodinger_eq(x:np.ndarray, 
                         potential_fun:Callable[[np.ndarray], np.ndarray],
                         h_bar:float = h_bar, 
                         m:float = m0, 
                         num_eigenvalues:Optional[int] = None,
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
    V = potential_fun(x)
    dx = x[1] - x[0]
    H = construct_hamiltonian_operator(V, dx = dx, h_bar = h_bar, m = m, dtype = np.float64)
    E, Psis = solve_eigenvalue(H, num_eigenvalues = num_eigenvalues)

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

def plot_energy_level(x:np.ndarray, E:np.ndarray, ax:Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    plot energy levels
    
    Parameters
    ----------
    x : grid points
    E : energy eigenvalues
    
    """
    
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

        ax.text( x[-1], E[ii], f'E{ii}', fontsize=fontsize, color = color)
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
    plot_energy_level(x, E, ax=ax, **kwargs)

    ax.set_ylabel('')
    return ax

