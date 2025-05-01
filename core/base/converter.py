import numpy as np

from core.utility.constant import c


def nu_hz_to_lambda_cm(nu_hz: np.ndarray) -> np.ndarray:
    """
    Convert frequency in Hz to wavelength in cm.
    """
    return c / nu_hz


def lambda_cm_to_nu_hz(lambda_nm: np.ndarray) -> np.ndarray:
    """
    Convert wavelength in cm to frequency in Hz.
    """
    return c / lambda_nm
