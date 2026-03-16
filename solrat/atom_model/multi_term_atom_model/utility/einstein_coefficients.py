from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s


def b_ul_from_a_ul_multi_term_atom(a_ul_sm1: float, nu_ul: float) -> float:
    r"""
    Transform from :math:`A_{ul}` [s^-1] to :math:`B_{ul}` [cm^2/erg/s]

    Reference: (LL04 7.33)
    """
    factor = 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
    return a_ul_sm1 / factor


def b_lu_from_b_ul_multi_term_atom(b_ul: float, Lu: float, Ll: float) -> float:
    r"""
    Transform from :math:`B_{lu}` [cm^2/erg/s] to `B_{ul}` [cm^2/erg/s]

    Reference: (7.33)
    """
    return b_ul * (2 * Lu + 1) / (2 * Ll + 1)
