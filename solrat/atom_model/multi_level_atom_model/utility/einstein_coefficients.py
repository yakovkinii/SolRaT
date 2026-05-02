from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s
from solrat.engine.functions.decorators import log_function


@log_function
def b_ul_from_a_ul_multi_level_atom(a_ul_sm1: float, nu_ul: float) -> float:
    r"""
    Transform from :math:`A_{ul}` [s^-1] to :math:`B_{ul}` [cm^2/erg/s].

    Reference: (LL04 7.8)
    """
    factor = 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
    return a_ul_sm1 / factor


@log_function
def b_lu_from_b_ul_multi_level_atom(b_ul: float, Ju: float, Jl: float) -> float:
    r"""
    Transform from :math:`B_{ul}` to :math:`B_{lu}` using detailed balance:

    .. math::
        (2 J_l + 1) B(\alpha_l J_l \to \alpha_u J_u) = (2 J_u + 1) B(\alpha_u J_u \to \alpha_l J_l)

    Reference: (LL04 7.8)
    """
    return b_ul * (2 * Ju + 1) / (2 * Jl + 1)
