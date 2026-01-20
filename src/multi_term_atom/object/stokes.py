import numpy as np


class Stokes:
    def __init__(self, nu: np.ndarray, I: np.ndarray, Q: np.ndarray, U: np.ndarray, V: np.ndarray):  # noqa: E741
        """
        Container class for Stokes parameters (I, Q, U, V) as functions of frequency.

        :param nu: Frequency array [Hz]
        :param I: Stokes I parameter (intensity)
        :param Q: Stokes Q parameter (linear polarization)
        :param U: Stokes U parameter (linear polarization)
        :param V: Stokes V parameter (circular polarization)
        """

        self.nu = nu
        self.I = I  # noqa: E741
        self.Q = Q
        self.U = U
        self.V = V
