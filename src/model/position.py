import numpy as np


def get_1D_position_mask(x:np.ndarray, x_range:np.ndarray) -> np.ndarray:
    """ 
    get the mask of x in the range of x_range 
    
    Usage
    -----
    mask = get_1D_position_mask(x, x_range)
    
    Parameters
    ----------
    x : position
    x_range : range of position
    
    Returns
    -------
    mask : mask of x in the range of x_range
    
    """
    x_range = np.asarray(x_range)
    if x_range.ndim == 1:
        x_range = np.expand_dims(x_range, axis=0)
        
    mask = np.zeros_like(x, dtype=bool)
    for ii in range(len(x_range)):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_or(mask, np.bitwise_and(x >= x_min, x <= x_max))
    return mask

