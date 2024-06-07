import numpy as np


""" fermi dirac """
def fermi_dirac_distribution(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Fermi-Dirac distribution function """
    return 1 / (np.exp((E - Ef) / kT) + 1)
def d_fermi_dirac_distribution_dE(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Derivative of Fermi-Dirac distribution function with respect to energy """
    return -1 / (kT * (np.exp((E - Ef) / kT) + 1) ** 2)
def d_fermi_dirac_distribution_dEf(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Derivative of Fermi-Dirac distribution function with respect to Fermi energy """
    return 1 / (kT * (np.exp((E - Ef) / kT) + 1) ** 2)


""" bose einstein """
def bose_einstein_distribution(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Bose-Einstein distribution function """
    return 1 / (np.exp((E - Ef) / kT) - 1)
def d_bose_einstein_distribution_dE(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Derivative of Bose-Einstein distribution function with respect to energy """
    return -1 / (kT * (np.exp((E - Ef) / kT) - 1) ** 2)
def d_bose_einstein_distribution_dEf(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Derivative of Bose-Einstein distribution function with respect to Fermi energy """
    return 1 / (kT * (np.exp((E - Ef) / kT) - 1) ** 2)

""" maxwell boltzmann """
def maxwell_boltzmann_distribution(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Maxwell-Boltzmann distribution function """
    return np.exp(-(E - Ef) / kT)
def d_maxwell_boltzmann_distribution_dE(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Derivative of Maxwell-Boltzmann distribution function with respect to energy """
    return -1 / kT * np.exp(-(E - Ef) / kT)
def d_maxwell_boltzmann_distribution_dEf(E: np.ndarray, Ef: float, kT: float) -> np.ndarray:
    """ Derivative of Maxwell-Boltzmann distribution function with respect to Fermi energy """
    return 1 / kT * np.exp(-(E - Ef) / kT)



