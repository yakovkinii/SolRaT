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

    def eta_sI(self):
        return np.real(self.eta_rho_sI)

    def eta_sQ(self):
        return np.real(self.eta_rho_sQ)

    def eta_sU(self):
        return np.real(self.eta_rho_sU)

    def eta_sV(self):
        return np.real(self.eta_rho_sV)

    def K(self):
        # 6.85
        K = np.zeros((len(self.eta_rho_sV), 4, 4), dtype=np.float64)
        K[:, 0, 0] = -np.real(self.eta_rho_sI - self.eta_rho_aI)
        K[:, 0, 1] = -np.real(self.eta_rho_sQ - self.eta_rho_aQ)
        K[:, 0, 2] = -np.real(self.eta_rho_sU - self.eta_rho_aU)
        K[:, 0, 3] = -np.real(self.eta_rho_sV - self.eta_rho_aV)
        K[:, 1, 0] = K[:, 0, 1]
        K[:, 1, 1] = K[:, 0, 0]
        K[:, 1, 2] = -np.imag(self.eta_rho_sV - self.eta_rho_aV)
        K[:, 1, 3] = -np.imag(-self.eta_rho_sU + self.eta_rho_aU)
        K[:, 2, 0] = K[:, 0, 2]
        K[:, 2, 1] = -K[:, 1, 2]
        K[:, 2, 2] = K[:, 0, 0]
        K[:, 2, 3] = -np.imag(self.eta_rho_sQ - self.eta_rho_aQ)
        K[:, 3, 0] = K[:, 0, 3]
        K[:, 3, 1] = -K[:, 1, 3]
        K[:, 3, 2] = -K[:, 2, 3]
        K[:, 3, 3] = K[:, 0, 0]
        return K

    def get_eta_for_tau(self):
        return np.real(self.eta_rho_aI - self.eta_rho_sI).max() #[:, None, None] #.max()

    def K_tau(self):
        """
        K/eta
        scales to line tau which is taken as max absorption
        """
        return self.K() / self.get_eta_for_tau()

    def epsilon(self):
        epsilon = np.zeros((len(self.eta_rho_sV), 4, 1), dtype=np.float64)
        epsilon[:, 0, 0] = self.epsilonI
        epsilon[:, 1, 0] = self.epsilonQ
        epsilon[:, 2, 0] = self.epsilonU
        epsilon[:, 3, 0] = self.epsilonV
        return epsilon

    def epsilon_tau(self):
        """
        scales to 'continuum' which is taken as first frequency point
        """
        return self.epsilon() / self.get_eta_for_tau()

    def eta_tau(self):
        """
        scales to 'continuum' which is taken as first frequency point
        """
        return (np.real(self.eta_rho_aI - self.eta_rho_sI) / np.real(self.eta_rho_aI - self.eta_rho_sI)[len(self.eta_rho_sV)*3//4])[:,None,None]
