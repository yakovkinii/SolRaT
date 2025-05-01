import numpy as np

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
sqrt_pi = np.sqrt(np.pi)

h = 6.626196e-27  # erg s
c = 2.99792458e10  # cm/s
kB = 1.380658e-16
e_0 = 0.480321e-9
m_e = 9.109390e-28
atomic_mass_unit = 1.660539e-24
mu0 = h * e_0 / (4 * np.pi * m_e * c)


def energy_cmm1_to_frequency_hz(energy_cmm1: float) -> float:
    return c * energy_cmm1  # Hz



