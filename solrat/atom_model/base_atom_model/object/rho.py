from typing import TypeVar


class BaseRho:
    r"""
    Base class for the statistical tensor :math:`\rho^K_Q`
    """


RhoT = TypeVar("RhoT", bound=BaseRho)
