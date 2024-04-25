from numpy import exp

from core.utility.constant import h, c, kB


def get_BP(nu, T):
    return 2 * h * nu**3 / c**2 / (exp(h * nu / (kB * T) - 1))
