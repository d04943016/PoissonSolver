import numpy as np

from typing import Callable

""" gaussian density of state """
def gaussian_density_of_state(E: np.ndarray, E0: float, sigma: float, N:float = 1.0) -> np.ndarray:
    """ 
    Gaussian density of state 
    
    usage
    -----
    dos = gaussian_density_of_state(E, E0, sigma, N)

    parameters
    ----------
    E: Energy
    E0: Center of the gaussian
    sigma: Width of the gaussian
    N: total state number of the gaussian state

    returns
    -------
    dos: Density of state
    
    """
    return N/ (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((E - E0) / sigma) ** 2)

def d_gaussian_density_of_state_dE(E: np.ndarray, E0: float, sigma: float, N:float = 1.0) -> np.ndarray:
    """ 
    Derivative of Gaussian density of state with respect to energy
    
    usage
    -----
    d_dos_dE = d_gaussian_density_of_state_dE(E, E0, sigma, N)

    parameters
    ----------
    E: Energy
    E0: Center of the gaussian
    sigma: Width of the gaussian
    N: total state number of the gaussian state

    returns
    -------
    d_dos_dE: Derivative of density of state with respect to energy
    
    """
    return - (E - E0) / (sigma ** 2) * gaussian_density_of_state(E, E0, sigma, N)

def d_gaussian_density_of_state_dE0(E: np.ndarray, E0: float, sigma: float, N:float = 1.0) -> np.ndarray:
    """ 
    Derivative of Gaussian density of state with respect to center of the gaussian
    
    usage
    -----
    d_dos_dE0 = d_gaussian_density_of_state_dE0(E, E0, sigma, N)

    parameters
    ----------
    E: Energy
    E0: Center of the gaussian
    sigma: Width of the gaussian
    N: total state number of the gaussian state

    returns
    -------
    d_dos_dE0: Derivative of density of state with respect to center of the gaussian
    
    """
    return -d_gaussian_density_of_state_dE(E, E0, sigma, N)

""" numerical function """
def get_d_dos_dE_func(dos_func:Callable[[np.ndarray], np.ndarray], h=1e-6) -> Callable[[np.ndarray], np.ndarray]:
    """ 
    Get numerical derivative of density of state with respect to energy
    
    usage
    -----
    d_dos_dE_func = get_d_dos_dE_func(E, dos_func, h)

    parameters
    ----------
    E: Energy
    dos_func: Density of state function
    h: step size for numerical derivative

    returns
    -------
    d_dos_dE_func: Numerical derivative of density of state with respect to energy
    
    """
    def d_dos_dE_func(E: np.ndarray) -> np.ndarray:
        return (dos_func(E + h) - dos_func(E - h)) / (2 * h)
    return d_dos_dE_func

