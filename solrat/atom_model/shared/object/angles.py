class Angles:
    r"""
    A container for all angles defining the LOS and B directions.

    :param chi: LOS angle chi
    :param theta: LOS angle theta
    :param gamma: LOS angle gamma
    :param chi_B: B angle chi
    :param theta_B: B angle theta

    Reference: Fig. 5.9.
    """

    def __init__(self, chi: float = 0, theta: float = 0, gamma: float = 0, chi_B: float = 0, theta_B: float = 0):
        self.chi = chi
        self.theta = theta
        self.gamma = gamma
        self.chi_B = chi_B
        self.theta_B = theta_B

    def _key(self) -> tuple:
        return self.chi, self.theta, self.gamma, self.chi_B, self.theta_B

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other) -> bool:
        return isinstance(other, Angles) and self._key() == other._key()
