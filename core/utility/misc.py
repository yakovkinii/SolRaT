import logging

from core.utility.constant import mu0, h


def nu_larmor(magnetic_field_gauss: float) -> float:
    """
    νL = 1.3996×10^6 B, with B expressed in G and νL in s−1
    Reference: (3.10)
    """
    logging.warning("Running untested code")
    return magnetic_field_gauss * mu0 / h
