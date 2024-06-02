import numpy as np


Nc = {'Si':2.8e19,  'GaAs':4.7e17,  'Ge':1.04e19, 'AlAs':1.5e19, 'GaP':1.8e19, 'InAs':8.3e16, 'InP':5.2e17 }  # cm^-3
Nv = {'Si':1.04e19, 'GaAs':7.0e18,  'Ge':6.0e18,  'AlAs':1.7e19, 'GaP':1.9e19, 'InAs':6.4e18, 'InP':1.1e19 }  # cm^-3
Eg = {'Si':1.12,    'GaAs':1.43,    'Ge':0.66,    'AlAs':2.16,   'GaP':2.21,   'InAs':0.36,   'InP':1.35  }  # eV

# electron affinity
EA = {'Si':4.05,    'GaAs':4.07,    'Ge':4.0,     'AlAs':2.62,   'GaP':4.3,    'InAs':4.9,    'InP':4.35    }  # eV
epsilon_r = {'Si':11.7, 'GaAs':12.9,'Ge':16.0,    'AlAs':10.1,   'GaP':11.1,   'InAs':14.6,   'InP':12.4    }  # 1/cm

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



