class Angles:
    def __init__(self, chi: float = 0, theta: float = 0, gamma: float = 0, chi_B: float = 0, theta_B: float = 0):
        """
        A container for all angles defining the LOS and B directions.

        :param chi: LOS angle chi
        :param theta: LOS angle theta
        :param gamma: LOS angle gamma
        :param chi_B: B angle chi
        :param theta_B: B angle theta

        Reference: Fig. 5.9.
        """
        self.chi = chi
        self.theta = theta
        self.gamma = gamma
        self.chi_B = chi_B
        self.theta_B = theta_B
