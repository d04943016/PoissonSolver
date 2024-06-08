import numpy as np
from .distribution import fermi_dirac_distribution as f_FD
from .position import get_1D_position_mask

from typing import List, Callable

# constants / InGaN's value is an average value
Nc = {'Si':2.8e19,  'GaAs':4.7e17,  'Ge':1.04e19, 'AlAs':1.5e19, 'GaP':1.8e19, 'InAs':8.3e16, 'InP':5.2e17, 'GaN':2.3e18, 'InGaN':1.5e18  }  # cm^-3
Nv = {'Si':1.04e19, 'GaAs':7.0e18,  'Ge':6.0e18,  'AlAs':1.7e19, 'GaP':1.9e19, 'InAs':6.4e18, 'InP':1.1e19, 'GaN':1.0e19, 'InGaN':1.0e19  }  # cm^-3
Eg = {'Si':1.12,    'GaAs':1.43,    'Ge':0.66,    'AlAs':2.16,   'GaP':2.21,   'InAs':0.36,   'InP':1.35,   'GaN':3.4,    'InGaN':2.05    }  # eV

# electron affinity
EA = {'Si':4.05,    'GaAs':4.07,    'Ge':4.0,     'AlAs':2.62,   'GaP':4.3,    'InAs':4.9,    'InP':4.35,   'GaN':4.1,    'InGaN':4.9}  # eV
epsilon_r = {'Si':11.7, 'GaAs':12.9,'Ge':16.0,    'AlAs':10.1,   'GaP':11.1,   'InAs':14.6,   'InP':12.4,   'GaN':8.9,    'InGaN':12.0} 


""" basic semi-conductor """
def cal_electron_density(Ec:np.ndarray, Ef:float, kT:float, Nc:float = Nc['Si']) -> np.ndarray:
    """ 
    calculate electron density
    
    Parameters
    ----------
    Ec : the conduction band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    Nc : the electron density constant
    
    Returns
    -------
    n : the electron density

    """
    return Nc * np.exp( -(Ec - Ef) / kT )

def cal_hole_density(Ev:np.ndarray, Ef:float, kT:float, Nv:float = Nv['Si']) -> np.ndarray:
    """ 
    calculate hole density 
    
    Parameters
    ----------
    Ev : the valence band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    Nv : the hole density constant

    Returns
    -------
    p : the hole density

    """
    return Nv * np.exp( -(Ef - Ev) / kT )

def cal_dn_dEc(Ec:np.ndarray, Ef:float, kT:float, Nc:float = Nc['Si']) -> np.ndarray:
    """ 
    calculate dn_dEc (the derivative of electron density with respect to Ec)
    
    Parameters
    ----------
    Ec : the conduction band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    Nc : the electron density constant

    Returns
    -------
    dn_dEc : the derivative of electron density with respect to Ec

    """
    return - Nc * np.exp( -(Ec - Ef) / kT ) / kT

def cal_dp_dEv(Ev:np.ndarray, Ef:float, kT:float, Nv:float = Nv['Si']) -> np.ndarray:
    """ 
    calculate dp_dEv (the derivative of hole density with respect to Ev)
    
    Parameters
    ----------
    Ev : the valence band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    Nv : the hole density constant

    Returns
    -------
    dp_dEv : the derivative of hole density with respect to Ev

    """
    return Nv * np.exp( -(Ef - Ev) / kT ) / kT

def cal_Ev_from_Ec(Ec:np.ndarray, Eg:float) -> np.ndarray:
    """ 
    calculate Ev from Ec 
    
    Parameters
    ----------
    Ec : the conduction band energy edge
    Eg : the band gap energy 

    Returns
    -------
    Ev : the valence band energy edge

    """
    return Ec - Eg

def cal_Ec_from_Ev(Ev:np.ndarray, Eg:float) -> np.ndarray:
    """ 
    calculate Ec from Ev 
    
    Parameters
    ----------
    Ev : the valence band energy edge
    Eg : the band gap energy

    Returns
    -------
    Ec : the conduction band energy edge

    """
    return Ev + Eg

def dope_constant_donor(x:np.ndarray, Na:float, x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate constant donor doping 
    
    Parameters
    ----------
    x : the position
    Na : the donor doping
    x_range : the range of the donor doping, 
              (1) x_range = [x_min, x_max]
              (2) x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
    Returns
    -------
    Na_vec : the donor doping vector

    """ 
    mask = get_1D_position_mask(x, x_range)
    Na_vec = np.zeros_like(x)
    Na_vec[mask] = Na
    return Na_vec

def dope_constant_acceptor(x:np.ndarray, Nd:float, x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate constant acceptor doping 
    
    Parameters
    ----------
    x : the position
    Nd : the acceptor doping
    x_range : the range of the acceptor doping,
              (1) x_range = [x_min, x_max]
              (2) x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
    Returns
    -------
    Nd_vec : the acceptor doping vector

    """ 
    Nd_vec = dope_constant_donor(x = x, Na = Nd, x_range = x_range)
    return Nd_vec

""" more general case """
def cal_charge_density_from_dos_fun(dos_fun:Callable[[np.ndarray], np.ndarray], Ef:float, kT:float, E_min:float, E_max:float, pts = 1000, axis = 0, **kwargs) -> np.ndarray:
    """ 
    calculate charge density from dos_fun with Fermi-Dirac distribution 
    
    Parameters
    ----------
    dos_fun : the density of states function
    Ef : the Fermi level
    kT : the thermal energy
    E_min : the minimum energy of integration
    E_max : the maximum energy of integration
    pts : the number of points
    axis : the axis to integrate
    kwargs : the keyword arguments for dos_fun

    Returns
    -------
    charge_density : the charge density

    """
    E = np.linspace(E_min, E_max, pts)
    dos = dos_fun(E, **kwargs)
    prob = f_FD(E = E, Ef = Ef, kT = kT)
    return np.trapz(dos * prob, E, axis = axis)
def cal_d_charge_density_from_d_dos_fun(d_dos_fun:Callable[[np.ndarray], np.ndarray], Ef:float, kT:float, E_min:float, E_max:float, pts = 1000, axis = 0, **kwargs) -> np.ndarray:
    """ 
    calculate charge density derivative from d_dos_fun with Fermi-Dirac distribution
    
    Parameters
    ----------
    d_dos_fun : the derivative of density of states function
    Ef : the Fermi level
    kT : the thermal energy
    E_min : the minimum energy of integration
    E_max : the maximum energy of integration
    pts : the number of points
    axis : the axis to integrate
    kwargs : the keyword arguments for d_dos_fun

    Returns
    -------
    d_charge_density : the charge density derivative

    """
    E = np.linspace(E_min, E_max, pts)
    d_dos = d_dos_fun(E, **kwargs)
    prob = f_FD(E = E, Ef = Ef, kT = kT)
    return np.trapz(d_dos * prob, E, axis = axis)

""" 1D charge density """
def cal_electron_density_at_different_x(x:np.ndarray, Ec:np.ndarray, Ef:float, kT:float, materials:str, x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate electron density in different materials 
    
    Parameters
    ----------
    x : the position
    Ec : the conduction band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
    
    Returns
    -------
    n : the electron density

    """
    x_range = np.asarray(x_range)
    n = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        n[mask] = cal_electron_density(Ec = Ec[mask], Ef = Ef, kT = kT, Nc = Nc[material])
    return n

def cal_hole_density_at_different_x(x:np.ndarray, Ev:np.ndarray, Ef:float, kT:float, materials:str, x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate hole density in different materials 
    
    Parameters
    ----------
    x : the position
    Ev : the valence band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
    
    Returns
    -------
    p : the hole density

    """
    x_range = np.asarray(x_range)
    p = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        p[mask] = cal_hole_density(Ev = Ev[mask], Ef = Ef, kT = kT, Nv = Nv[material])
    return p

def cal_dn_dEc_at_different_x(x:np.ndarray, Ec:np.ndarray, Ef:float, kT:float, materials:str, x_range:np.ndarray) -> np.ndarray: 
    """ 
    calculate dn_dEc in different materials 
    
    Parameters
    ----------
    x : the position
    Ec : the conduction band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
    
    Returns
    -------
    dn_dEc : the derivative of electron density with respect to Ec

    """
    x_range = np.asarray(x_range)
    dn_dEc = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        dn_dEc[mask] = cal_dn_dEc(Ec = Ec[mask], Ef = Ef, kT = kT, Nc = Nc[material])
    return dn_dEc

def cal_dp_dEv_at_different_x(x:np.ndarray, Ev:np.ndarray, Ef:float, kT:float, materials:str, x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate dp_dEv in different materials 
    
    Parameters
    ----------
    x : the position
    Ev : the valence band energy edge
    Ef : the Fermi level
    kT : the thermal energy
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
    
    Returns
    -------
    dp_dEv : the derivative of hole density with respect to Ev

    """
    x_range = np.asarray(x_range)
    dp_dEv = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        dp_dEv[mask] = cal_dp_dEv(Ev = Ev[mask], Ef = Ef, kT = kT, Nv = Nv[material])
    return dp_dEv

def cal_Ev_from_Ec_at_different_x(x:np.ndarray, Ec:np.ndarray, materials:List[str], x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate Ev from Ec in different materials 
    
    Parameters
    ----------
    x : the position
    Ec : the conduction band energy edge
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
    
    Returns
    -------
    Ev : the valence band energy edge

    """
    x_range = np.asarray(x_range)
    Ev = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        Ev[mask] = cal_Ev_from_Ec(Ec = Ec[mask], Eg = Eg[material])
    return Ev

def cal_Ec_from_Ev_at_different_x(x:np.ndarray, Ev:np.ndarray, materials:List[str], x_range:np.ndarray) -> np.ndarray:
    """ 
    calculate Ec from Ev in different materials 
    
    Parameters
    ----------
    x : the position
    Ev : the valence band energy edge
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
              
    Returns
    -------
    Ec : the conduction band energy edge
    
    """
    x_range = np.asarray(x_range)
    Ec = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        Ec[mask] = cal_Ec_from_Ev(Ev = Ev[mask], Eg = Eg[material])
    return Ec

def get_episolon_r_at_different_x(x:np.ndarray, materials:List[str], x_range:np.ndarray) -> np.ndarray:
    """ 
    get epsilon_r (relative permittivity) in different materials 
    
    Parameters
    ----------
    x : the position
    materials : the materials
    x_range : the range of the materials,
              * x_range = [[x_min1, x_max1], [x_min2, x_max2], ...]
              * [x_min1, x_max1] is the range of the first material
              * [x_min_n, x_max_n] is the range of the n-th material
    
    Returns
    -------
    epsilon_r_vec : the epsilon_r vector
    
    """
    x_range = np.asarray(x_range)
    epsilon_r_vec = np.zeros_like(x)
    if len(materials) != len(x_range):
        raise ValueError(f"The length of materials ({len(materials)}) and x_range ({len(x_range)}) should be the same.") 

    for ii, material in enumerate(materials):
        x_min, x_max = np.min(x_range[ii]), np.max(x_range[ii])
        mask = np.bitwise_and(x >= x_min, x <= x_max)
        epsilon_r_vec[mask] = epsilon_r[material]
    return epsilon_r_vec

""" 1D field """
def cal_electric_field(x:np.ndarray, V:np.ndarray) -> np.ndarray:
    """ 
    calculate electric field 
    
    Parameters
    ----------
    x : the position
    V : the potential energy

    Returns
    -------
    E : the electric field

    """
    E = -np.gradient(V, x)
    return E

""" from total charge density to source function """
def to_poisson_source(charge_density:np.ndarray, epsilon:np.ndarray) -> np.ndarray:
    """ 
    calculate source function 
    
    Parameters
    ----------
    charge_density : total charge density
    epsilon : the permittivity

    Returns
    -------
    src : the source function

    """
    src = - (1/epsilon) * charge_density
    return src

def to_poisson_source_function(x:np.ndarray, V:np.ndarray, get_charge_density_fun:Callable[[np.ndarray, np.ndarray], np.ndarray], get_epsilon_fun:Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """ 
    calculate source function 
    
    Parameters
    ----------
    x : the position
    V : the potential energy
    get_charge_density_fun : the function to get charge density
    get_epsilon_fun : the function to get epsilon

    Returns
    -------
    src : the source function

    """
    charge_density = get_charge_density_fun(x, V)
    epsilon = get_epsilon_fun(x)
    src = to_poisson_source(charge_density = charge_density, epsilon = epsilon)
    return src






