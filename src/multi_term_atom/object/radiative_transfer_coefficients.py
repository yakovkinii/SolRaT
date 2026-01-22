from typing import Tuple

import numpy as np


class RadiativeTransferCoefficients:
    def __init__(
        self,
        eta_rho_aI,
        eta_rho_aQ,
        eta_rho_aU,
        eta_rho_aV,
        eta_rho_sI,
        eta_rho_sQ,
        eta_rho_sU,
        eta_rho_sV,
        epsilonI,
        epsilonQ,
        epsilonU,
        epsilonV,
    ):
        """
        This is a container class for all radiative transfer coefficients.
        """
        self.eta_rho_aI = eta_rho_aI
        self.eta_rho_aQ = eta_rho_aQ
        self.eta_rho_aU = eta_rho_aU
        self.eta_rho_aV = eta_rho_aV
        self.eta_rho_sI = eta_rho_sI
        self.eta_rho_sQ = eta_rho_sQ
        self.eta_rho_sU = eta_rho_sU
        self.eta_rho_sV = eta_rho_sV
        self.epsilonI = epsilonI
        self.epsilonQ = epsilonQ
        self.epsilonU = epsilonU
        self.epsilonV = epsilonV
        self._eta_tau_scale = self._compute_eta_tau_scale()

    def split_eta_rho(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split real and imaginary parts of eta_rho for constructing the propagation matrix K
        """
        etaI = np.real(self.eta_rho_aI - self.eta_rho_sI)
        etaQ = np.real(self.eta_rho_aQ - self.eta_rho_sQ)
        etaU = np.real(self.eta_rho_aU - self.eta_rho_sU)
        etaV = np.real(self.eta_rho_aV - self.eta_rho_sV)

        rhoQ = np.imag(self.eta_rho_aQ - self.eta_rho_sQ)
        rhoU = np.imag(self.eta_rho_aU - self.eta_rho_sU)
        rhoV = np.imag(self.eta_rho_aV - self.eta_rho_sV)
        return etaI, etaQ, etaU, etaV, rhoQ, rhoU, rhoV

    def K_z(self) -> np.ndarray:
        """
        RT matrix K related to the z-propagation equation:
        .. math::
            dStokes/dz = - K * Stokes + epsilon - K_continuum Stokes + epsilon_continuum
        .. math::
            K = K_A - K_S
        If N = 1 was used (no atom concentration provided), then the RTE code computes primed Kʹ and epsilonʹ:
        .. math::
            Kʹ = K /  N
        .. math::
            epsilonʹ = epsilon / N
        The transfer equation then becomes:
        .. math::
            dStokes/dz = - Kʹ * N * Stokes + epsilonʹ * N [ - K_continuum Stokes + epsilon_continuum ]
        Reference: (6.83-6.85)
        """
        etaI, etaQ, etaU, etaV, rhoQ, rhoU, rhoV = self.split_eta_rho()

        # shape: [len(nu), 4, 4]
        K = np.empty((len(etaI), 4, 4), dtype=np.float64)
        K[:, 0, 0] = etaI
        K[:, 0, 1] = etaQ
        K[:, 0, 2] = etaU
        K[:, 0, 3] = etaV
        K[:, 1, 0] = etaQ
        K[:, 1, 1] = etaI
        K[:, 1, 2] = rhoV
        K[:, 1, 3] = -rhoU
        K[:, 2, 0] = etaU
        K[:, 2, 1] = -rhoV
        K[:, 2, 2] = etaI
        K[:, 2, 3] = rhoQ
        K[:, 3, 0] = etaV
        K[:, 3, 1] = rhoU
        K[:, 3, 2] = -rhoQ
        K[:, 3, 3] = etaI
        return K

    def K_tau(self) -> np.ndarray:
        """
        RT matrix K related to the tau-propagation equation:

        .. math::
            dStokes/dtau_line = - K_tau_line * Stokes + epsilon_tau_line - Stokes / eta_LC + BP(T) eI / eta_LC
        where

        .. math::
            eta_LC = eta_line / eta_continuum
        In terms of dtau_continuum we then have:

        .. math::
            dStokes/dtau_continuum = - K_tau_line * Stokes * eta_LC + epsilon_tau_line * eta_LC - Stokes + BP(T) eI
        With a boundary condition of

        .. math::
            Stokes[tau->+inf] -> BP(T0)
        We can normalize the Stokes on BP(T0):

        .. math::
            dStokes/dtau_continuum = - K_tau_line * Stokes * eta_LC + epsilon_tau_line * eta_LC / BP(T0)
                                     - Stokes + BP(T)/BP(T0) eI
        With a boundary condition of

        .. math::
            Stokes[tau->+inf] -> 1
        """
        K = self.K_z()
        return K / self._eta_tau_scale

    def _compute_eta_tau_scale(self) -> np.ndarray:
        """
        Line-center eta, assuming a single spectral line
        """
        return np.max(np.abs(np.real(self.eta_rho_aI - self.eta_rho_sI)))

    def epsilon_z(self) -> np.ndarray:
        """
        RT matrix emission coefficient epsilon related to the z-propagation equation: see the comments for K_z()
        Reference: (6.83-6.85)
        """
        # shape: [len(nu), 4, 1]
        epsilon = np.zeros((len(self.eta_rho_sV), 4, 1), dtype=np.float64)
        epsilon[:, 0, 0] = self.epsilonI
        epsilon[:, 1, 0] = self.epsilonQ
        epsilon[:, 2, 0] = self.epsilonU
        epsilon[:, 3, 0] = self.epsilonV
        return epsilon

    def epsilon_tau(self) -> np.ndarray:
        """
        RT matrix emission coefficient epsilon related to the tau-propagation equation: see the comments for K_tau()
        Reference: (6.83-6.85)
        """
        return self.epsilon_z() / self._eta_tau_scale
