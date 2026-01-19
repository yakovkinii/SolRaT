"""
TODO
TODO  This file needs improved documentation.
TODO
"""

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

    def eta_sI(self):
        return np.real(self.eta_rho_sI)

    def eta_sQ(self):
        return np.real(self.eta_rho_sQ)

    def eta_sU(self):
        return np.real(self.eta_rho_sU)

    def eta_sV(self):
        return np.real(self.eta_rho_sV)

    def _split_eta_rho(self):
        etaI = np.real(self.eta_rho_aI)
        etaQ = np.real(self.eta_rho_aQ)
        etaU = np.real(self.eta_rho_aU)
        etaV = np.real(self.eta_rho_aV)

        rhoQ = np.imag(self.eta_rho_aQ)
        rhoU = np.imag(self.eta_rho_aU)
        rhoV = np.imag(self.eta_rho_aV)
        return etaI, etaQ, etaU, etaV, rhoQ, rhoU, rhoV

    def K_nu(self):
        # Build K(ν) per-frequency with LL04 layout.
        etaI, etaQ, etaU, etaV, rhoQ, rhoU, rhoV = self._split_eta_rho()

        # Stack into [Nν, 4, 4]
        K = np.empty((len(etaI), 4, 4), dtype=np.float64)
        K[:, 0, 0] = etaI; K[:, 0, 1] = etaQ; K[:, 0, 2] = etaU; K[:, 0, 3] = etaV
        K[:, 1, 0] = etaQ; K[:, 1, 1] = etaI; K[:, 1, 2] = rhoV; K[:, 1, 3] = -rhoU
        K[:, 2, 0] = etaU; K[:, 2, 1] = -rhoV;K[:, 2, 2] = etaI; K[:, 2, 3] = rhoQ
        K[:, 3, 0] = etaV; K[:, 3, 1] = rhoU; K[:, 3, 2] = -rhoQ;K[:, 3, 3] = etaI
        return K

    def K_tau(self):
        # Rescale by the same scalar used for ε and ητ (see #8/#9 below)
        K = self.K_nu()
        return K / self._eta_tau_scale

    def _compute_eta_tau_scale(self):
        # Use the *intensity* channel of absorptive minus emissive kernels.
        x = np.real(self.eta_rho_aI - self.eta_rho_sI)  # shape [Nν]
        maxabs = np.max(np.abs(x))
        return maxabs # Todo check
        maxval = np.max(x)
        # Floor protects against near-cancellation or negative maxima.
        floor = max(1e-12 * maxabs, 1e-20)
        return max(maxval, floor)

    def epsilon(self):
        # returns [Nν, 4, 1]
        epsilon = np.zeros((len(self.eta_rho_sV), 4, 1), dtype=np.float64)
        epsilon[:, 0, 0] = self.epsilonI
        epsilon[:, 1, 0] = self.epsilonQ
        epsilon[:, 2, 0] = self.epsilonU
        epsilon[:, 3, 0] = self.epsilonV
        return epsilon

    def epsilon_tau(self):
        # just scale the already-built vector
        return self.epsilon() / self._eta_tau_scale

    def eta_tau(self):
        x = np.real(self.eta_rho_aI - self.eta_rho_sI)  # [Nν]
        return x / self._eta_tau_scale
