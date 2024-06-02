
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Callable, Optional

""" gradient """
def do_gradient(vec:np.ndarray, dx:float = 1.0) -> np.ndarray:
    return (vec[1:] - vec[:-1]) / dx

""" Laplacian """
def construct_Laplacian_operator(size:int, dx:float = 1.0, reduced:bool = True, dtype = np.float64) -> np.ndarray:
    if reduced:
        A = np.ones( (size, 3), dtype = dtype )
        A[:,1] = -2.0
        A = A/(dx**2)
    else:
        A = np.zeros( (size, size), dtype = dtype )
        dx_2 = dx**2
        diagonal = -2.0 / dx_2
        off_diagonal = 1.0 / dx_2
        A[0,0], A[0,1] = diagonal, off_diagonal
        A[-1,-2], A[-1,-1] = off_diagonal, diagonal
        for ii in range(1, size-1):
            A[ii,ii-1] = off_diagonal
            A[ii,ii]   = diagonal
            A[ii,ii+1] = off_diagonal
    return A

def do_Laplacian(vec:np.ndarray, v0:float = 0.0, vn:float = 0.0, dx:float = 1.0) -> np.ndarray:
    dv_dxdx = np.zeros_like(vec)
    dv_dxdx[1:-1] = vec[:-2] - 2.0*vec[1:-1] + vec[2:]
    dv_dxdx[0]  = v0 - 2.0*vec[0] + vec[1] 
    dv_dxdx[-1] = vec[-2] - 2.0*vec[-1] + vn

    return dv_dxdx / (dx**2)

def solve_Laplacian(A:np.ndarray, b:np.ndarray, reduced:bool = True) -> np.ndarray:
    """ solve Ax = b, A is a Laplcian matrix """
    if not reduced:
        return np.linalg.solve(A, b)
    A, b = A.copy(), b.copy()
    for ii in range(A.shape[0]-1):
        ratio = A[ii+1,0]/A[ii,1]
        # A[ii+1, 0] = 0.0
        A[ii+1, 1] -= ratio*A[ii,2]
        b[ii+1] -= ratio*b[ii]

    for ii in range(A.shape[0]-1, 0, -1):
        ratio = A[ii-1,2]/A[ii,1]
        # A[ii-1, 2] = 0.0
        b[ii-1] -= ratio*b[ii]
    return b / A[:,1]

""" Updater """
class AdamUpdater:
    def __init__(self, size:int, lr:float=0.1, beta1:float = 0.9, beta2:float = 0.999, epsilon:float = 1e-8):
        self.size = size
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.init()

    def init(self):
        self.vt    = np.zeros(self.size)
        self.sigma = np.zeros(self.size)
        self.step = 0

    def update(self, vec:np.ndarray, delta:np.ndarray):
        if vec.shape != delta.shape:
            raise ValueError("shape mismatch")
        
        self.step += 1
        self.vt    = self.beta1*self.vt    + (1.0-self.beta1)*delta
        self.sigma = self.beta2*self.sigma + (1.0-self.beta2)*(delta**2)
        
        vt_hat    =    self.vt / (1.0 - self.beta1**self.step)
        sigma_hat = self.sigma / (1.0 - self.beta2**self.step)
        new_delta = vt_hat / (np.sqrt(sigma_hat) + self.epsilon)
        return vec + self.lr*new_delta
    
    def __call__(self, vec:np.ndarray, delta:np.ndarray):
        return self.update(vec, delta)
    
    @property
    def size(self):
        return self.__size
    
    @size.setter
    def size(self, size:int):
        self.__size = size
        self.init()

class Adagrad:
    def __init__(self, size:int, lr:float=0.5, epsilon:float = 1e-8):
        self.size = size
        self.lr = lr
        self.epsilon = epsilon
        self.init()

    def init(self):
        self.sigma = np.zeros(self.size)
        self.step = 0

    def update(self, vec:np.ndarray, delta:np.ndarray):
        if vec.shape != delta.shape:
            raise ValueError("shape mismatch")
        
        self.step += 1
        self.sigma += delta**2
        new_delta = delta / np.sqrt(self.sigma  + self.epsilon)
        return vec + self.lr*new_delta
    
    def __call__(self, vec:np.ndarray, delta:np.ndarray):
        return self.update(vec, delta)
    
    @property
    def size(self):
        return self.__size
    
    @size.setter
    def size(self, size:int):
        self.__size = size
        self.init()

""" Poisson Solver """
def construct_updater(size, V_updater:Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None):
    if V_updater is None:
        V_updater = lambda vec, delta: vec + delta  
    elif callable(V_updater):
        V_updater = V_updater
    elif V_updater.upper() == 'ADAM':
        V_updater = AdamUpdater(size = size)
    elif V_updater.upper() == 'ADAGRAD':
        V_updater = Adagrad(size = size)
    else:
        V_updater = AdamUpdater(size = size)
    return V_updater
def _construct_V_init(x:np.array, **kwargs):
    if 'V_init' in kwargs:
        V = kwargs['V_init']
    elif 'V_init_func' in kwargs:
        V = kwargs['V_init_func'](x)
    else:
        V = np.zeros_like(x)
    return V
def solve_Poisson_1D(x:np.ndarray, 
                     src_fun:Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] ,
                     dsrc_dV_fun:Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] ,
                     v0:Optional[float] = None, 
                     vn:Optional[float] = None,
                     V_updater:Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                     max_iter:int = 1000,
                     tol:float = 1e-6,
                     progress_bar:bool = True, 
                     boundary_condition:Optional[Callable[[np.ndarray,np.ndarray], np.ndarray]] = None,
                     record:bool = False,
                     **kwargs
    ) -> np.ndarray:
    """ solve Poisson equation """
    # initialize
    V = _construct_V_init(x, **kwargs)
    record_V = []
    if record:
        record_V.append(V.copy())

    # construct updater
    V_updater = construct_updater(size = x.size, V_updater = V_updater)

    # iteration
    dx = x[1] - x[0]
    iterator = trange(max_iter) if progress_bar else range(max_iter)
    for step in iterator:
        # calculate rho
        src = src_fun(x, V) 
        dsrc_dV = dsrc_dV_fun(x, V)

        # calculate Laplacian
        A = construct_Laplacian_operator(x.size, dx = dx)
        A[:,1] -= dsrc_dV

        # calculate b
        v0_tmp = V[0] if v0 is None else v0
        vn_tmp = V[-1] if vn is None else vn
        BC = boundary_condition(x, V) if boundary_condition is not None else np.zeros_like(V)
        b = src - do_Laplacian(V, v0 = v0_tmp, vn = vn_tmp, dx = dx) - BC

        # solve
        delta_V = solve_Laplacian(A, b)
        
        # check convergence
        V = V_updater(V, delta_V)

        # record
        if record:
            record_V.append(V.copy())

        if np.linalg.norm(delta_V) < tol:
            break
    
    if record:
        return V, np.array( record_V )
    return V    


""" plots """
def plot_Poisson_1D(x:np.ndarray, 
                    V:np.ndarray = None, 
                    n:np.ndarray = None, 
                    p:np.ndarray = None, 
                    total_charge_density:np.ndarray = None,
                    Ec:np.ndarray = None, 
                    Ev:np.ndarray = None, 
                    Field:np.ndarray = None, 
                    Na:np.ndarray = None, 
                    Nd:np.ndarray = None,
    **kwargs):
    fig, ax = plt.subplots(1, 4, figsize=(30, 6))
    if Ec is not None:
        ax[0].plot(x * 1e4, Ec, label='Ec')
    if Ev is not None:
        ax[0].plot(x * 1e4, Ev, label='Ev')

    ax[0].plot(x * 1e4, np.zeros_like(x), 'k--', label='Ef')
    ax[0].legend()
    ax[0].set_xlabel(r'$x (\mu m)$')
    ax[0].set_ylabel(r'Energy (eV)')
    ax[0].set_title(r'Band Diagram')

    if Na is not None:
        ax[1].plot(x * 1e4, Na, label='Na')
    if Nd is not None:
        ax[1].plot(x * 1e4, Nd, label='Nd')
    if n is not None:
        ax[1].plot(x * 1e4, n, label='n')
    if p is not None:
        ax[1].plot(x * 1e4, p, label='p')
    ax[1].legend()
    ax[1].set_xlabel(r'$x (\mu m)$')
    ax[1].set_ylabel(r'$Density (1/cm^{-3})$')
    ax[1].set_title(r'Charge Density')
    ax[1].set_yscale('log')
    
    if total_charge_density is not None:
        mask = total_charge_density > 0
        total_mask = np.zeros_like(total_charge_density)
        total_mask[mask] = total_charge_density[mask]
        ax[2].fill_between(x * 1e4, total_mask, color='lightblue', label='Positive Charge (+)')
        mask = total_charge_density < 0
        total_mask = np.zeros_like(total_charge_density)
        total_mask[mask] =  np.abs( total_charge_density[mask] )
        ax[2].fill_between(x * 1e4, total_mask, color='pink', label='Negative Charge (-)')
        
        ax[2].set_xlabel(r'$x (\mu m)$')
        ax[2].set_ylabel(r'$Density (1/cm^{-3})$')
        ax[2].set_title(r'Total Charge Density')
        ax[2].set_yscale('log')
        ax[2].legend()

    if Field is not None:
        ax[3].plot( (x[1:]+x[:-1] )/2* 1e4, Field, label='Field')
    ax[3].legend()
    ax[3].set_xlabel(r'$x (\mu m)$')
    ax[3].set_ylabel(r'Field (V/cm)')
    ax[3].set_title(r'Electric Field & Voltage')

    if V is not None:
        ax3_2 = ax[3].twinx()
        ax3_2.plot(x * 1e4, V, 'r', label='V')
        ax3_2.set_ylabel(r'Voltage (V)', color='red', rotation=-90, labelpad=20)  # Set the y-axis label rotation to 90 degrees and increase the labelpad value to move the label further away from the axis
        ax3_2.tick_params(axis='y', colors='r')  # Set the tick color to red
        ax3_2.legend()

    return ax




