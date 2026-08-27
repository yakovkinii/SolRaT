import numpy as np

from solrat.atom_model.shared.utility.allen import i0_allen
from solrat.atom_model.shared.utility.functions import get_planck_BP


class Stokes:
    """
    Container class for Stokes parameters (I, Q, U, V) as function of frequency.

    :param nu: Frequency array [Hz]
    :param I: Stokes I parameter (intensity)
    :param Q: Stokes Q parameter (linear polarization)
    :param U: Stokes U parameter (linear polarization)
    :param V: Stokes V parameter (circular polarization)
    """

    def __init__(self, nu: np.ndarray, I: np.ndarray, Q: np.ndarray, U: np.ndarray, V: np.ndarray):  # noqa: E741
        self.nu = nu
        self.I = I  # noqa: E741
        self.Q = Q
        self.U = U
        self.V = V

    @classmethod
    def from_BP(cls, nu_sm1: np.ndarray, temperature_K: float) -> "Stokes":
        """
        Get the Stokes profiles from Planck's distribution.

        :param nu_sm1: frequencies [1/s]
        :param temperature_K: temperature in K
        :return: Stokes instance
        """
        return cls(
            nu=nu_sm1,
            I=get_planck_BP(nu_sm1=nu_sm1, temperature_K=temperature_K),
            Q=nu_sm1 * 0,
            U=nu_sm1 * 0,
            V=nu_sm1 * 0,
        )

    @classmethod
    def from_zeros(cls, nu_sm1: np.ndarray) -> "Stokes":
        """
        Get the Stokes profiles that are zeros.

        :param nu_sm1: frequencies [1/s]
        :return: Stokes instance
        """
        return cls(
            nu=nu_sm1,
            I=nu_sm1 * 0,
            Q=nu_sm1 * 0,
            U=nu_sm1 * 0,
            V=nu_sm1 * 0,
        )


def get_boundary_stokes_allen(nu_sm1: np.ndarray, lambda_A: float, mu: float) -> Stokes:
    """
    Get the unpolarized Allen continuum boundary Stokes vector.

    :param nu_sm1: frequencies [1/s]
    :param lambda_A: wavelength used for the Allen continuum intensity [Angstrom]
    :param mu: heliocentric cosine of the line of sight
    :return: Stokes instance
    """
    return Stokes(
        nu=nu_sm1,
        I=np.full_like(nu_sm1, i0_allen(lambda_A=lambda_A, mu=mu)),
        Q=nu_sm1 * 0,
        U=nu_sm1 * 0,
        V=nu_sm1 * 0,
    )
