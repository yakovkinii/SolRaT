import logging
from typing import Union

import numpy as np
from numpy import exp

from src.core.physics.constants import c, h, kB, mu0


def get_BP(nu: np.ndarray, T: float) -> np.ndarray:
    logging.warning("get_BP needs additional testing")
    return 2 * h * nu**3 / c**2 / (exp(h * nu / (kB * T) - 1))


def nu_larmor(magnetic_field_gauss: np.ndarray) -> np.ndarray:
    """
    νL = 1.3996×10^6 B, with B expressed in G and νL in s−1
    Reference: (3.10)
    """
    return magnetic_field_gauss * mu0 / h


def energy_cmm1_to_frequency_hz(energy_cmm1: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return c * energy_cmm1


def frequency_hz_to_energy_cmm1(frequency_hz: np.ndarray) -> np.ndarray:
    return frequency_hz / c


def frequency_hz_to_lambda_cm(frequency_hz: np.ndarray) -> np.ndarray:
    return c / frequency_hz


def lambda_cm_to_frequency_hz(lambda_nm: np.ndarray) -> np.ndarray:
    return c / lambda_nm
