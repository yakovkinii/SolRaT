from src.core.physics.constants import c, h


def b_ul_from_a_two_term_atom(a_ul, nu_ul):
    """
    Reference: (7.32)
    """
    factor = 2 * h / c**2 * nu_ul**3
    return a_ul / factor


def b_lu_from_b_ul_two_term_atom(b_ul, Lu, Ll):
    """
    Reference: (7.32)
    """
    return b_ul * (2 * Lu + 1) / (2 * Ll + 1)
