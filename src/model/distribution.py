import numpy as np


""" fermi dirac """
def fermi_dirac_distribution(E: np.ndarray, kT: float, Ef: float) -> np.ndarray:
    """ 
    Fermi-Dirac distribution function 
    
    Usage
    -----
    f_FD = fermi_dirac_distribution(E, Ef, kT)

    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)

    Returns
    -------
    f_FD : Fermi-Dirac distribution probability

    """
    return 1 / (np.exp((E - Ef) / kT) + 1)
def d_fermi_dirac_distribution_dE(E: np.ndarray, kT: float, Ef: float) -> np.ndarray:
    """ 
    Derivative of Fermi-Dirac distribution function with respect to energy 
    
    Usage
    -----
    df_dE = d_fermi_dirac_distribution_dE(E, Ef, kT)

    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)

    Returns
    -------
    df_dE : Derivative of Fermi-Dirac distribution probability with respect to energy

    """
    return -1 / (kT * (np.exp((E - Ef) / kT) + 1) ** 2)
def d_fermi_dirac_distribution_dEf(E: np.ndarray, kT: float, Ef: float) -> np.ndarray:
    """ 
    Derivative of Fermi-Dirac distribution function with respect to Fermi energy 
    
    Usage
    -----
    df_dEf = d_fermi_dirac_distribution_dEf(E, Ef, kT)

    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)

    Returns
    -------
    df_dEf : Derivative of Fermi-Dirac distribution probability with respect to Fermi energy

    """
    return -d_fermi_dirac_distribution_dE(E=E, kT=kT, Ef=Ef)

""" bose einstein """
def bose_einstein_distribution(E: np.ndarray, kT: float, Ef: float) -> np.ndarray:
    """ 
    Bose-Einstein distribution function 
    
    Usage
    -----
    f_BE = bose_einstein_distribution(E, Ef, kT)

    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)

    Returns
    -------
    f_BE : Bose-Einstein distribution probability

    """
    return 1 / (np.exp((E - Ef) / kT) - 1)
def d_bose_einstein_distribution_dE(E: np.ndarray, kT: float, Ef: float) -> np.ndarray:
    """ 
    Derivative of Bose-Einstein distribution function with respect to energy 
    
    Usage
    -----
    df_dE = d_bose_einstein_distribution_dE(E, Ef, kT)

    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)

    Returns
    -------
    df_dE : Derivative of Bose-Einstein distribution probability with respect to energy

    """
    return -1 / (kT * (np.exp((E - Ef) / kT) - 1) ** 2)
def d_bose_einstein_distribution_dEf(E: np.ndarray, kT: float, Ef: float) -> np.ndarray:
    """ 
    Derivative of Bose-Einstein distribution function with respect to Fermi energy 
    
    Usage
    -----
    df_dEf = d_bose_einstein_distribution_dEf(E, Ef, kT)
    
    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)
    
    Returns
    -------
    df_dEf : Derivative of Bose-Einstein distribution probability with respect to Fermi energy
    
    """
    return -d_bose_einstein_distribution_dE(E=E, kT=kT, Ef=Ef)

""" maxwell boltzmann """
def maxwell_boltzmann_distribution(E: np.ndarray, kT: float, Ef: float = 0.0) -> np.ndarray:
    """ 
    Maxwell-Boltzmann distribution function 
    
    Usage
    -----
    f_MB = maxwell_boltzmann_distribution(E, Ef, kT)
    
    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)
    
    Returns
    -------
    f_MB : Maxwell-Boltzmann distribution probability
    
    """
    return np.exp(-(E - Ef) / kT)
def d_maxwell_boltzmann_distribution_dE(E: np.ndarray, kT: float, Ef: float = 0.0) -> np.ndarray:
    """ 
    Derivative of Maxwell-Boltzmann distribution function with respect to energy 
    
    Usage
    -----
    df_dE = d_maxwell_boltzmann_distribution_dE(E, Ef, kT)
    
    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)

    Returns
    -------
    df_dE : Derivative of Maxwell-Boltzmann distribution probability with respect to energy

    """
    return -1 / kT * np.exp(-(E - Ef) / kT)
def d_maxwell_boltzmann_distribution_dEf(E: np.ndarray, kT: float, Ef: float = 0.0) -> np.ndarray:
    """ 
    Derivative of Maxwell-Boltzmann distribution function with respect to Fermi energy 
    
    Usage
    -----
    df_dEf = d_maxwell_boltzmann_distribution_dEf(E, Ef, kT)
    
    Parameters
    ----------
    E : energy
    Ef : Fermi energy
    kT : Boltzmann constant times temperature (thermal energy)
    
    Returns
    -------
    df_dEf : Derivative of Maxwell-Boltzmann distribution probability with respect to Fermi energy
    
    """
    return -d_maxwell_boltzmann_distribution_dE(E=E, kT=kT, Ef=Ef)



