import numpy as np
from numpy import exp

from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, kB_erg_Km1, mu0_erg_gaussm1
from solrat.engine.functions.decorators import log_function
from solrat.engine.functions.special import FloatOrNDArrayT


def get_planck_BP(nu_sm1: FloatOrNDArrayT, temperature_K: float) -> FloatOrNDArrayT:
    """
    Planck function
    Reference: (below 5.40)
    """
    return 2 * h_erg_s * nu_sm1**3 / c_cm_sm1**2 / (exp(h_erg_s * nu_sm1 / kB_erg_Km1 / temperature_K) - 1)


def nu_larmor(magnetic_field_gauss: np.ndarray) -> np.ndarray:
    """
    Larmor frequency in Hz
    Reference: (3.10)
    """
    return magnetic_field_gauss * mu0_erg_gaussm1 / h_erg_s


def energy_cmm1_to_frequency_sm1(energy_cmm1: FloatOrNDArrayT) -> FloatOrNDArrayT:
    """
    Convert energy to frequency
    """
    return c_cm_sm1 * energy_cmm1


def lambda_cm_to_frequency_sm1(lambda_cm: FloatOrNDArrayT) -> FloatOrNDArrayT:
    """
    Convert wavelength to frequency
    """
    return c_cm_sm1 / lambda_cm


def lambda_A_to_frequency_sm1(lambda_A: FloatOrNDArrayT) -> FloatOrNDArrayT:
    """
    Convert wavelength to frequency
    """
    return c_cm_sm1 * 1e8 / lambda_A


def frequency_sm1_to_lambda_A(frequency_sm1: FloatOrNDArrayT) -> FloatOrNDArrayT:
    """
    Convert frequency to wavelength
    """
    return c_cm_sm1 / frequency_sm1 * 1e8


def energy_cmm1_to_erg(energy_cmm1: float) -> float:
    """
    Convert energy units
    """
    return h_erg_s * c_cm_sm1 * energy_cmm1


def lambda_vacuum_to_air(lambda_vacuum_A: FloatOrNDArrayT) -> FloatOrNDArrayT:
    """
    Convert vacuum wavelength to air wavelength (Angstrom).
    Uses the Edlen (1966) formula, valid for ~2000-10000 A.
    Reference: 10.1088/0026-1394/2/2/002
    """
    sigma2 = (1e4 / lambda_vacuum_A) ** 2
    n = 1 + (8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2)) * 1e-8
    return lambda_vacuum_A / n


def lambda_air_to_vacuum(lambda_air_A: FloatOrNDArrayT, n_iter: int = 3) -> FloatOrNDArrayT:
    """
    Convert air wavelength to vacuum wavelength (Angstrom).
    Inverts Edlen (1966) vacuum-to-air relation by fixed-point iteration.
    Reference: 10.1088/0026-1394/2/2/002
    """
    lambda_vac = lambda_air_A  # initial guess

    for _ in range(n_iter):
        sigma2 = (1e4 / lambda_vac) ** 2
        n = 1 + (8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2)) * 1e-8
        lambda_vac = lambda_air_A * n

    return lambda_vac


@log_function
def get_frequencies_from_air_wavelength_range(
    lower_wavelength_A: float, upper_wavelength_A: float, step_A: float
) -> np.ndarray:
    """
    Build a frequency array (Hz) from a vacuum wavelength range (Angstrom).
    """
    lambda_A_air = np.arange(lower_wavelength_A, upper_wavelength_A, step_A)
    lambda_A_vac = lambda_air_to_vacuum(lambda_A_air)
    return lambda_A_to_frequency_sm1(lambda_A_vac)
