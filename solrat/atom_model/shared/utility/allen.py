import numpy as np

f"""
Allen's :math:`J^K_Q` continuum intensities. 

Reference: HAZEL, A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
"""

C_CM_SM1 = 2.99792458e10  # PC, speed of light [cm/s]
H_ERG_S = 6.62606885e-27  # PH, Planck constant [erg s]
RSUN_ARCSEC = 976.6  # RSUN [arcsec]

ALLEN_IC_LAMBDA_MICRON = np.array(
    [
        0.200000, 0.220000, 0.240000, 0.260000, 0.280000, 0.300000, 0.320000, 0.340000, 0.360000,
        0.370000, 0.380000, 0.390000, 0.400000, 0.410000, 0.420000, 0.430000, 0.440000, 0.450000,
        0.460000, 0.480000, 0.500000, 0.550000, 0.600000, 0.650000, 0.700000, 0.750000, 0.800000,
        0.900000, 1.00000, 1.10000, 1.20000, 1.40000, 1.60000, 1.80000, 2.00000, 2.50000, 3.00000,
        4.00000, 5.00000, 6.00000, 8.00000, 10.0000, 12.0000,
    ]
)
ALLEN_IC_ILAMBDA = np.array(
    [
        0.0600000, 0.210000, 0.290000, 0.600000, 1.30000, 2.45000, 3.25000, 3.77000, 4.13000,
        4.23000, 4.63000, 4.95000, 5.15000, 5.26000, 5.28000, 5.24000, 5.19000, 5.10000, 5.00000,
        4.79000, 4.55000, 4.02000, 3.52000, 3.06000, 2.69000, 2.28000, 2.03000, 1.57000, 1.26000,
        1.01000, 0.810000, 0.530000, 0.360000, 0.238000, 0.160000, 0.0780000, 0.0410000, 0.0142000,
        0.00620000, 0.00320000, 0.000950000, 0.000350000, 0.000180000,
    ]
)
ALLEN_CL_LAMBDA_MICRON = np.array(
    [
        0.200000, 0.220000, 0.245000, 0.265000, 0.280000, 0.300000, 0.320000, 0.350000, 0.370000,
        0.380000, 0.400000, 0.450000, 0.500000, 0.550000, 0.600000, 0.800000, 1.00000, 1.50000,
        2.00000, 3.00000, 5.00000, 10.0000,
    ]
)
ALLEN_CL_U1 = np.array(
    [
        0.120000, -1.30000, -0.100000, -0.100000, 0.380000, 0.740000, 0.880000, 0.980000, 1.03000,
        0.920000, 0.910000, 0.990000, 0.970000, 0.930000, 0.880000, 0.730000, 0.640000, 0.570000,
        0.480000, 0.350000, 0.220000, 0.150000,
    ]
)
ALLEN_CL_U2 = np.array(
    [
        0.330000, 1.60000, 0.850000, 0.900000, 0.570000, 0.200000, 0.0300000, -0.100000, -0.160000,
        -0.0500000, -0.0500000, -0.170000, -0.220000, -0.230000, -0.230000, -0.220000, -0.200000,
        -0.210000, -0.180000, -0.120000, -0.0700000, -0.0700000,
    ]
)

IC_INU = ALLEN_IC_ILAMBDA * ALLEN_IC_LAMBDA_MICRON**2 / (C_CM_SM1 * 1.0e4)
IC_LAMBDA_A = ALLEN_IC_LAMBDA_MICRON * 1.0e4
CL_LAMBDA_A = ALLEN_CL_LAMBDA_MICRON * 1.0e4
I0_INU = ALLEN_IC_ILAMBDA * 1.0e14 * (IC_LAMBDA_A * 1.0e-8) ** 2 / C_CM_SM1


def i0_allen(lambda_A: float, mu: float) -> float:
    r"""
    The Allen continuum intensity

    :param lambda_A: wavelength [Angstrom].
    :param mu: heliocentric cosine of the line of sight.
    :return: continuum Stokes-I intensity.
    """
    if mu == 0:
        return 0.0

    u1 = float(np.interp(lambda_A, CL_LAMBDA_A, ALLEN_CL_U1))
    u2 = float(np.interp(lambda_A, CL_LAMBDA_A, ALLEN_CL_U2))
    i0 = float(np.interp(lambda_A, IC_LAMBDA_A, I0_INU))
    return (1.0 - u1 - u2 + u1 * mu + u2 * mu**2) * i0


def geometric_factors(height_arcsec: float):
    r"""
    The height-dilution geometry factors :math:`(a_0, a_1, a_2)` for :math:`J` and
    :math:`(b_0, b_1, b_2)` for :math:`K` at a height above the surface, from the solid angle
    subtended by the Sun. At the surface (``height_arcsec = 0``), limb-darkening-free limits are taken.
    """
    if height_arcsec != 0.0:
        sg = RSUN_ARCSEC / (height_arcsec + RSUN_ARCSEC)
        cg = np.sqrt(1.0 - sg**2)
        a0 = 1.0 - cg
        a1 = cg - 0.5 - 0.5 * cg**2 / sg * np.log((1.0 + sg) / cg)
        a2 = (cg + 2.0) * (cg - 1.0) / (3.0 * (cg + 1.0))
        b0 = (1.0 - cg**3) / 3.0
        b1 = (8.0 * cg**3 - 3.0 * cg**2 - 2.0) / 24.0 - cg**4 / (8.0 * sg) * np.log((1.0 + sg) / cg)
        b2 = (cg - 1.0) * (3.0 * cg**3 + 6.0 * cg**2 + 4.0 * cg + 2.0) / (15.0 * (cg + 1.0))
        return a0, a1, a2, b0, b1, b2
    return 1.0, -0.5, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 12.0, -2.0 / 15.0


def nbar_allen(lambda_A: float, height_arcsec: float, reduction_factor: float = 1.0) -> float:
    r"""
    The nbar parameter of the prescribed radiation tensor at a wavelength and height above the
    surface, from Allen's continuum intensity and centre-to-limb coefficients (Hazel ``nbar_allen``).

    :param lambda_A: wavelength [Angstrom].
    :param height_arcsec: height above the surface [arcsec] (1'' = 725 km).
    :param reduction_factor: multiplies J^0_0; 1.0 keeps Allen's value.
    :return: nbar.
    """
    intensity = float(np.interp(lambda_A, IC_LAMBDA_A, IC_INU))
    u1 = float(np.interp(lambda_A, CL_LAMBDA_A, ALLEN_CL_U1))
    u2 = float(np.interp(lambda_A, CL_LAMBDA_A, ALLEN_CL_U2))
    a0, a1, a2, _, _, _ = geometric_factors(height_arcsec)
    j_field = 0.5 * intensity * (a0 + a1 * u1 + a2 * u2)
    return 1.0e10 * (lambda_A**3 * 1.0e-24 / (2.0 * H_ERG_S * C_CM_SM1)) * j_field * reduction_factor


def omega_allen(lambda_A: float, height_arcsec: float, reduction_factor: float = 1.0) -> float:
    r"""
    The omega anisotropy parameter of the prescribed radiation tensor at a wavelength and height
    above the surface, from Allen's data (Hazel ``omega_allen``): :math:`w = (3K - J)/(2J)`.

    :param lambda_A: wavelength [Angstrom].
    :param height_arcsec: height above the surface [arcsec].
    :param reduction_factor: multiplies omega; 1.0 keeps Allen's value.
    :return: omega
    """
    intensity = float(np.interp(lambda_A, IC_LAMBDA_A, IC_INU))
    u1 = float(np.interp(lambda_A, CL_LAMBDA_A, ALLEN_CL_U1))
    u2 = float(np.interp(lambda_A, CL_LAMBDA_A, ALLEN_CL_U2))
    a0, a1, a2, b0, b1, b2 = geometric_factors(height_arcsec)
    j_field = 0.5 * intensity * (a0 + a1 * u1 + a2 * u2)
    k_field = 0.5 * intensity * (b0 + b1 * u1 + b2 * u2)
    return (3.0 * k_field - j_field) / (2.0 * j_field) * reduction_factor
