import numpy as np
from .distribution import fermi_dirac_distribution as f_FD
from .position import get_1D_position_mask

from typing import List, Callable

Nc = {'Si':2.8e19,  'GaAs':4.7e17,  'Ge':1.04e19, 'AlAs':1.5e19, 'GaP':1.8e19, 'InAs':8.3e16, 'InP':5.2e17 }  # cm^-3
Nv = {'Si':1.04e19, 'GaAs':7.0e18,  'Ge':6.0e18,  'AlAs':1.7e19, 'GaP':1.9e19, 'InAs':6.4e18, 'InP':1.1e19 }  # cm^-3
Eg = {'Si':1.12,    'GaAs':1.43,    'Ge':0.66,    'AlAs':2.16,   'GaP':2.21,   'InAs':0.36,   'InP':1.35  }  # eV

# electron affinity
EA = {'Si':4.05,    'GaAs':4.07,    'Ge':4.0,     'AlAs':2.62,   'GaP':4.3,    'InAs':4.9,    'InP':4.35    }  # eV
epsilon_r = {'Si':11.7, 'GaAs':12.9,'Ge':16.0,    'AlAs':10.1,   'GaP':11.1,   'InAs':14.6,   'InP':12.4    }  # 1/cm


""" basic semi-conductor """
def cal_electron_density(Ec:np.ndarray, Ef:float, kT:float, Nc:float = Nc['Si']) -> np.ndarray:
    """ calculate electron density """
    return Nc * np.exp( -(Ec - Ef) / kT )

def cal_hole_density(Ev:np.ndarray, Ef:float, kT:float, Nv:float = Nv['Si']) -> np.ndarray:
    """ calculate hole density """
    return Nv * np.exp( -(Ef - Ev) / kT )

def cal_dn_dEc(Ec:np.ndarray, Ef:float, kT:float, Nc:float = Nc['Si']) -> np.ndarray:
    """ calculate dn_dEc """
    return - Nc * np.exp( -(Ec - Ef) / kT ) / kT

def cal_dp_dEv(Ev:np.ndarray, Ef:float, kT:float, Nv:float = Nv['Si']) -> np.ndarray:
    """ calculate dp_dEv """
    return Nv * np.exp( -(Ef - Ev) / kT ) / kT

def cal_Ev_from_Ec(Ec:np.ndarray, Eg:float) -> np.ndarray:
    """ calculate Ev from Ec """
    return Ec - Eg

def cal_Ec_from_Ev(Ev:np.ndarray, Eg:float) -> np.ndarray:
    """ calculate Ec from Ev """
    return Ev + Eg

def dope_constant_donor(x:np.ndarray, Na:float, x_range:np.ndarray) -> np.ndarray:
    """ calculate constant donor doping """ 
    mask = get_1D_position_mask(x, x_range)
    Na_vec = np.zeros_like(x)
    Na_vec[mask] = Na
    return Na_vec

def dope_constant_acceptor(x:np.ndarray, Nd:float, x_range:np.ndarray) -> np.ndarray:
    """ calculate constant acceptor doping """ 
    Nd_vec = dope_constant_donor(x = x, Na = Nd, x_range = x_range)
    return Nd_vec

""" more general case """
def cal_charge_density_from_dos_fun(dos_fun:Callable[[np.ndarray], np.ndarray], Ef:float, kT:float, E_min:float, E_max:float, pts = 1000, axis = 0, **kwargs) -> np.ndarray:
    """ calculate charge density from dos_fun with Fermi-Dirac distribution """
    E = np.linspace(E_min, E_max, pts)
    dos = dos_fun(E, **kwargs)
    prob = f_FD(E = E, Ef = Ef, kT = kT)
    return np.trapz(dos * prob, E, axis = axis)
def cal_d_charge_density_from_d_dos_fun(d_dos_fun:Callable[[np.ndarray], np.ndarray], Ef:float, kT:float, E_min:float, E_max:float, pts = 1000, axis = 0, **kwargs) -> np.ndarray:
    """ calculate charge density derivative from d_dos_fun with Fermi-Dirac distribution"""
    E = np.linspace(E_min, E_max, pts)
    d_dos = d_dos_fun(E, **kwargs)
    prob = f_FD(E = E, Ef = Ef, kT = kT)
    return np.trapz(d_dos * prob, E, axis = axis)

""" 1D charge density """
def cal_electron_density_at_different_x(x:np.ndarray, Ec:np.ndarray, Ef:float, kT:float, materials:str, x_range:np.ndarray) -> np.ndarray:
    """ calculate electron density in different materials """
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
    """ calculate hole density in different materials """
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
    """ calculate dn_dEc in different materials """
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
    """ calculate dp_dEv in different materials """
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
    """ calculate Ev from Ec in different materials """
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
    """ calculate Ec from Ev in different materials """
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
    """ get epsilon_r in different materials """
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
    """ calculate electric field """
    E = -np.gradient(V, x)
    return E

""" from total charge density to source function """
def to_poisson_source(charge_density:np.ndarray, epsilon:np.ndarray) -> np.ndarray:
    """ calculate source function """
    src = - (1/epsilon) * charge_density
    return src






