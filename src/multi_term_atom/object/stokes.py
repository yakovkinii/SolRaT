import numpy as np


class Stokes:
    """
    Container class for Stokes parameters (I, Q, U, V) as functions of frequency.

    Attributes:
        nu: Frequency array [Hz]
        I: Stokes I parameter (intensity)
        Q: Stokes Q parameter (linear polarization)
        U: Stokes U parameter (linear polarization)
        V: Stokes V parameter (circular polarization)
    """

    def __init__(self, nu: np.ndarray, I: np.ndarray, Q: np.ndarray, U: np.ndarray, V: np.ndarray):
        self.nu = nu
        self.I = I
        self.Q = Q
        self.U = U
        self.V = V

    def __str__(self):
        return f"Stokes(nu: {len(self.nu)} points, I: [{self.I.min():.3e}, {self.I.max():.3e}])"

    def copy(self):
        """Create a deep copy of the Stokes vector"""
        return Stokes(
            nu=self.nu.copy(),
            I=self.I.copy(),
            Q=self.Q.copy(),
            U=self.U.copy(),
            V=self.V.copy()
        )

    def normalize(self, reference='I_max'):
        """
        Normalize Stokes parameters.

        Args:
            reference: Normalization reference ('I_max', 'I_continuum', or float value)
        """
        if reference == 'I_max':
            norm = self.I.max()
        elif reference == 'I_continuum':
            # Use average of first and last points as continuum
            norm = 0.5 * (self.I[0] + self.I[-1])
        else:
            norm = float(reference)

        return Stokes(
            nu=self.nu,
            I=self.I / norm,
            Q=self.Q / norm,
            U=self.U / norm,
            V=self.V / norm
        )
