
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Callable, Optional, Union

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

class AdagradUpdater:
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
    elif V_updater.upper() in ('ADAM', 'ADAMUPDATER'):
        V_updater = AdamUpdater(size = size)
    elif V_updater.upper() in ('ADAGRAD', 'ADAGRADUPDATER'):
        V_updater = AdagradUpdater(size = size)
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


""" plots 1D """
def plot_BandDiagram_1D(x:np.ndarray,
                        Ec:Optional[np.ndarray] = None,
                        Ev:Optional[np.ndarray] = None,
                        Ef:Optional[Union[float,np.ndarray]] = 0.0,
                        ax:Optional[plt.Axes] = None,
                        **kwargs,
    ) -> plt.Axes:
    """ plot band diagram """
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))
    
    # controls
    xscaling = kwargs.get('xscaling', 1e4)
    xlabel = kwargs.get('xlabel', r'$x (\mu m)$')
    ylabel = kwargs.get('ylabel', r'Energy (eV)')
    title = kwargs.get('title',   r'Band Diagram')
    fontsize = kwargs.get('fontsize', 12)
    linewidth = kwargs.get('linewidth', 2)

    # plot Ec 
    if Ec is not None:
        ax.plot(x * xscaling, Ec, linewidth=linewidth, label=kwargs.get('Ec_label', 'Ec'), color=kwargs.get('Ec_color', 'b'))
    
    # plot Ev
    if Ev is not None:
        ax.plot(x * xscaling, Ev, linewidth=linewidth, label=kwargs.get('Ev_label', 'Ev'), color=kwargs.get('Ev_color', 'orange'))

    # plot Ef
    if Ef is not None:
        Ef_linestyle = kwargs.get('Ef_linestyle', 'k--')
        if isinstance(Ef, (int, float)):
            ax.plot(x * xscaling, np.ones_like(x)*Ef, Ef_linestyle, linewidth=linewidth, label = kwargs.get('Ef_label', 'Ef'))
        else:
            ax.plot(x * xscaling, Ef, Ef_linestyle, linewidth=linewidth, label = kwargs.get('Ef_label', 'Ef'))

    # set controls
    ax.legend(fontsize = fontsize)
    ax.set_xlabel(xlabel, fontsize = fontsize)  
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title,   fontsize = fontsize)
    return ax

def plot_gaussian_Dit(x:np.ndarray,
                      E_dit_mean:np.ndarray,
                      E_dit_sigma:np.ndarray,
                      z:Optional[np.ndarray] = None,
                      ax:Optional[plt.Axes] = None,
                      **kwargs,
    ) -> plt.Axes:
    """ plot Dit """
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))
    
    # controls
    xscaling = kwargs.get('xscaling', 1e4)
    xlabel = kwargs.get('xlabel', r'$x (\mu m)$')
    ylabel = kwargs.get('ylabel', r'Energy (eV)')
    title = kwargs.get('title',   r'Band Diagram')
    fontsize = kwargs.get('fontsize', 12)
    linewidth = kwargs.get('linewidth', 2)
    alpha = kwargs.get('alpha', 0.3)

    # plot Dit
    z = 3 if z is None else z
    ax.plot(x * xscaling, E_dit_mean, linewidth=linewidth, label=kwargs.get('Dit_label', 'Dit'), color=kwargs.get('Dit_color', 'r'))
    ax.fill_between(x * xscaling, E_dit_mean - z*E_dit_sigma, E_dit_mean + z*E_dit_sigma, alpha=alpha, color=kwargs.get('fill_color', 'red'))

    # set controls
    ax.legend(fontsize = fontsize)
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title,   fontsize = fontsize)
    return ax

def plot_carrier_density_1D( x:np.ndarray, 
                             n:Optional[np.ndarray] = None,
                             p:Optional[np.ndarray] = None,
                             Na:Optional[np.ndarray] = None,
                             Nd:Optional[np.ndarray] = None,
                             Dit:Optional[np.ndarray] = None,
                             ax:Optional[plt.Axes] = None,
                             **kwargs,
    )->plt.Axes:
    """ plot charge density """
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    # controls
    xscaling = kwargs.get('xscaling', 1e4)
    xlabel = kwargs.get('xlabel', r'$x (\mu m)$')
    ylabel = kwargs.get('ylabel', r'Density $(1/cm^{-3})$')
    title = kwargs.get('title',   r'Charge Density')
    fontsize = kwargs.get('fontsize', 12)
    yscale = kwargs.get('yscale', 'log')
    linewidth = kwargs.get('linewidth', 2)

    # plot Na
    if Na is not None:
        ax.plot(x * xscaling, Na, linewidth=linewidth, label=kwargs.get('Na_label', 'Na'), color=kwargs.get('Na_color', 'b'))

    # plot Nd   
    if Nd is not None:
        ax.plot(x * xscaling, Nd, linewidth=linewidth, label=kwargs.get('Nd_label', 'Nd'), color=kwargs.get('Nd_color', 'orange'))

    # plot n
    if n is not None:
        ax.plot(x * xscaling, n, linewidth=linewidth, label=kwargs.get('n_label', 'n'), color=kwargs.get('n_color', 'green'))
    
    # plot p
    if p is not None:
        ax.plot(x * xscaling, p, linewidth=linewidth, label=kwargs.get('p_label', 'p'), color=kwargs.get('p_color', 'red'))

    # plot Dit
    if Dit is not None:
        ax.plot(x * xscaling, np.abs(Dit), linewidth=linewidth, label=kwargs.get('Dit_label', 'Dit'), color=kwargs.get('Dit_color', 'purple'))

    # set controls
    ax.legend(fontsize = fontsize)
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title,   fontsize = fontsize)
    ax.set_yscale(yscale)
    return ax

def plot_total_charge_density_1D(x:np.ndarray,
                                 total_charge_density:Optional[np.ndarray] = None,
                                 ax:Optional[plt.Axes] = None,
                                 **kwargs,
    )->plt.Axes:
    """ plot total charge density """
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    # controls
    xscaling = kwargs.get('xscaling', 1e4)
    xlabel = kwargs.get('xlabel', r'$x (\mu m)$')
    ylabel = kwargs.get('ylabel', r'Density $(1/cm^{-3})$')
    title = kwargs.get('title',   r'Total Charge Density')
    fontsize = kwargs.get('fontsize', 12)
    yscale = kwargs.get('yscale', 'log')

    positive_color = kwargs.get('positive_color', 'lightblue')
    negative_color = kwargs.get('negative_color', 'pink')

    # plot total charge density
    if total_charge_density is not None:
        mask = total_charge_density > 0
        total_mask = np.zeros_like(total_charge_density)
        total_mask[mask] = total_charge_density[mask]
        ax.fill_between(x * xscaling, total_mask, color=positive_color, label=kwargs.get('positive_label', 'Positive Charge (+)'))

        mask = total_charge_density < 0
        total_mask = np.zeros_like(total_charge_density)
        total_mask[mask] =  np.abs( total_charge_density[mask] )
        ax.fill_between(x * xscaling, total_mask, color=negative_color, label=kwargs.get('negative_label', 'Negative Charge (-)'))
        
    # set controls
    ax.legend(fontsize = fontsize)
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title,   fontsize = fontsize)
    ax.set_yscale(yscale)
    return ax

def plot_Field_1D(x:np.ndarray,
                  V:Optional[np.ndarray] = None,
                  Field:Optional[np.ndarray] = None,
                  ax:Optional[plt.Axes] = None,
                  **kwargs,
    )->plt.Axes:
    """ plot electric field """
    # get ax
    if ax is None:
        im_width = kwargs.get('im_width', 8)
        im_height = kwargs.get('im_height', 6)
        _, ax = plt.subplots(1, 1, figsize=(im_width, im_height))

    # controls
    xscaling = kwargs.get('xscaling', 1e4)
    fontsize = kwargs.get('fontsize', 12)
    linewidth = kwargs.get('linewidth', 2)
    rotation = 0
    labelpad = 0

    ax.set_title(kwargs.get('title',   r'Field'), fontsize = fontsize)

    # plot Field
    if Field is not None:
        ax.plot( x * xscaling, Field, label=kwargs.get('Field_label', 'Field'), linewidth=linewidth, color = kwargs.get('Field_color', 'b') )
        ax.legend(fontsize = fontsize)
        ax.set_xlabel(kwargs.get('xlabel', r'$x (\mu m)$'), fontsize = fontsize)
        ax.set_ylabel(kwargs.get('ylabel', r'Field (V/cm)'), fontsize = fontsize)
        
        if V is not None:
            ax = ax.twinx()
            rotation = -90
            labelpad = 20

    # plot V
    if V is not None:
        ax.plot(x * xscaling, V, label=kwargs.get('V_label', 'V'), linewidth=linewidth, color = kwargs.get('V_color', 'r') )
        ax.set_ylabel(r'Voltage (V)', color = kwargs.get('V_color', 'r'), rotation=rotation, labelpad=labelpad, fontsize = fontsize)
        ax.tick_params(axis='y', colors=kwargs.get('V_color', 'r'))
        ax.legend()
    return ax
        
def plot_Poisson_1D(x:np.ndarray, 
                    V:np.ndarray = None, 
                    n:np.ndarray = None, 
                    p:np.ndarray = None, 
                    total_charge_density:np.ndarray = None,
                    Ec:np.ndarray = None, 
                    Ev:np.ndarray = None, 
                    Ef:Optional[Union[float,np.ndarray]] = 0.0,
                    Field:np.ndarray = None, 
                    Na:np.ndarray = None, 
                    Nd:np.ndarray = None,
                    Dit:Optional[np.ndarray] = None,
    **kwargs):
    _, ax = plt.subplots(1, 4, figsize=(30, 6))

    # plot band diagram
    plot_BandDiagram_1D(x=x,Ec=Ec,Ev=Ev,Ef=Ef,ax=ax[0],**kwargs)

    # plot charge density
    plot_carrier_density_1D(x=x,n=n,p=p,Na=Na,Nd=Nd,Dit=Dit,ax=ax[1],**kwargs)

    # plot total charge density
    plot_total_charge_density_1D(x=x,total_charge_density=total_charge_density,ax=ax[2],**kwargs)

    # plot field
    plot_Field_1D(x=x,V=V,Field=Field,ax=ax[3],**kwargs)
    return ax




