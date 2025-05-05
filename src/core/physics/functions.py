from typing import Union

import numpy as np
from numpy import exp

from src.core.physics.constants import c_cm_sm1, h_erg_s, kB_erg_Km1, mu0_erg_gaussm1


def get_planck_BP(nu_sm1: np.ndarray, T_K: float) -> np.ndarray:
    """
    Planck function
    Reference: (below 5.40)
    """
    return 2 * h_erg_s * nu_sm1**3 / c_cm_sm1**2 / (exp(h_erg_s * nu_sm1 / (kB_erg_Km1 * T_K) - 1))


def nu_larmor(magnetic_field_gauss: np.ndarray) -> np.ndarray:
    """
    Larmor frequency in Hz
    Reference: (3.10)
    """
    return magnetic_field_gauss * mu0_erg_gaussm1 / h_erg_s


def energy_cmm1_to_frequency_hz(energy_cmm1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return c_cm_sm1 * energy_cmm1


def lambda_cm_to_frequency_hz(lambda_cm: np.ndarray) -> np.ndarray:
    return c_cm_sm1 / lambda_cm
