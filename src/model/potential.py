import numpy as np

from .position import get_1D_position_mask

from typing import Union

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

def triangle_potential(x:np.ndarray, x_range:np.ndarray, V0:float=0.0, V1:Union[float,np.ndarray]=-1.0, ratio:Union[float,np.ndarray]=0.5):
    """
    Triangle potential
    
    Usage
    -----
    V = triangle_potential(x, x_range, V0, V1)
    
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
    V = np.ones_like(x) * V0

    # x_range to ndarray
    x_range = np.asarray(x_range)
    if x_range.ndim == 1:
        x_range = np.expand_dims(x_range, axis=0)
    
    for x_range_ii in x_range:
        x_min, x_max = np.min(x_range_ii), np.max(x_range_ii)

        tmp_ratio = ratio if isinstance(ratio, (float, int)) else ratio[0]
        tmp_ratio = np.clip(np.abs(tmp_ratio), 0.0, 1.0)
        x_mid = x_min + tmp_ratio* (x_max - x_min)
        
        mask1 = np.bitwise_and(x >= x_min, x <= x_mid)
        mask2 = np.bitwise_and(x > x_mid, x <= x_max)

        v1 = V1 if isinstance(V1, (float, int)) else V1[0]

        V[mask1] += V0 + (v1 - V0) * (x[mask1] - x_min) / (x_mid - x_min)
        V[mask2] += v1 + (V0 - v1) * (x[mask2] - x_mid) / (x_max - x_mid)

    return V



