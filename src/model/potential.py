import numpy as np

from .position import get_1D_position_mask

def simple_harmonic_oscillator_potential(x:np.ndarray, k:float=1.0, x0:float=0.0):
    """
    Simple harmonic oscillator potential
    
    Usage
    -----
    V = simple_harmonic_oscillator_potential(x, k, x0)
    
    Parameters
    ----------
    x : position
    k : spring constant
    x0 : equilibrium position
    
    Returns
    -------
    V : potential energy
    
    """
    return 0.5 * k * (x - x0)**2

def finite_wells_potential(x:np.ndarray, x_range:np.ndarray, V0:float=0.0, V1:float=-1.0):
    """
    Finite wells potential
    
    Usage
    -----
    V = finite_wells_potential(x, x_range, V0, V1)
    
    Parameters
    ----------
    x : position
    x_range : range of position
    V0 : potential energy outside the range
    V1 : potential energy inside the range
    
    Returns
    -------
    V : potential energy
    
    """
    mask = get_1D_position_mask(x=x, x_range=x_range)
    V = np.ones_like(x) * V0
    V[mask] = V1
    return V


