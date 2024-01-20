from core.utility.constant import h, c


def b_ul_from_a_ul_two_level_atom(a_ul, nu):
    """
    real B(alpha_u J_u -> alpha_l J_l)

    nu - transition frequency

    Reference: (7.8)
    """
    factor = 2 * h / c**2 * nu**3
    return a_ul / factor


def a_ul_from_b_ul_two_level_atom(b_ul, nu):
    """
    real A(alpha_u J_u -> alpha_l J_l)

    nu - transition frequency

    Reference: (7.8)
    """
    factor = 2 * h / c**2 * nu**3
    return b_ul * factor


def b_lu_from_b_ul_two_level_atom(b_ul, j_u, j_l):
    """
    real B(alpha_l J_l -> alpha_u J_u)

    Reference: (7.8)
    """
    return b_ul * (2 * j_u + 1) / (2 * j_l + 1)


def b_ul_from_b_lu_two_level_atom(b_lu, j_u, j_l):
    """
    real B(alpha_l J_l -> alpha_u J_u)

    Reference: (7.8)
    """
    return b_lu * (2 * j_l + 1) / (2 * j_u + 1)
