from src.core.physics.constants import c_cm_sm1, h_erg_s


def b_ul_from_a_ul_two_term_atom(a_ul_sm1, nu_ul):
    """
    Aul [s^-1] to Bul [cm^2/erg/s]
    Reference: (7.33)
    """
    factor = 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
    return a_ul_sm1 / factor


def b_lu_from_b_ul_two_term_atom(b_ul, Lu, Ll):
    """
    Reference: (7.33)
    """
    return b_ul * (2 * Lu + 1) / (2 * Ll + 1)
