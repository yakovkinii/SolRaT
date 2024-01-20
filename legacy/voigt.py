import logging

import numpy as np
from numpy import pi, exp
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator


class Voigt:
    def __init__(self, coarse=False):
        print("Pre-computing Voigt profiles")
        if coarse:
            self.a_values = 10 ** (np.linspace(-1, -3, 5))
        else:
            self.a_values = 10 ** (np.linspace(0, -3, 50))
        self.a_values_lookup = {val: i for i, val in enumerate(self.a_values)}
        self.v_values = np.arange(-20, 20, 0.2)
        self.min_a = min(self.a_values)
        self.max_a = max(self.a_values)
        self.min_v = min(self.v_values)
        self.max_v = max(self.v_values)
        self.voigt_h = self.pre_compute_H()
        self.voigt_l = self.pre_compute_L()
        print("Finished pre-computing Voigt profiles")

    def pre_compute_H(self):
        result = [[exp(-(v**2)) for v in self.v_values]]
        for a in self.a_values[1:]:
            result_a = []
            for v in self.v_values:

                def integrand_H(y):
                    return a / pi * np.exp(-(y**2)) / ((v - y) ** 2 + a**2)

                integral = quad(integrand_H, -np.inf, np.inf)[0]
                result_a.append(integral)
            result.append(result_a)
        return RegularGridInterpolator(
            (self.a_values, self.v_values),
            np.array(result),
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )
        # return interp2d(self.v_values, self.a_values,  np.array(result), kind='cubic')

    # ValueError: When on a regular grid with x.size = m and y.size = n, if z.ndim == 2, then z must have shape (n, m)
    def pre_compute_L(self):
        result = []
        for a in self.a_values:
            result_a = []
            for v in self.v_values:

                def integrand_L(y):
                    return (
                        1 / pi * np.exp(-(y**2)) * (v - y) / ((v - y) ** 2 + a**2)
                    )

                integral = quad(integrand_L, -np.inf, np.inf)[0]
                result_a.append(integral)
            result.append(result_a)

        return RegularGridInterpolator(
            (self.a_values, self.v_values),
            np.array(result),
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )
        # return interp2d(self.v_values, self.a_values, np.array(result), kind='cubic')

    def get(self, v, a, profile: str):
        assert profile in ["H", "L"]

        a_v = np.column_stack((a, v))
        if profile == "H":
            result = self.voigt_h(a_v)
        else:
            result = self.voigt_l(a_v)

        if np.any(a >= self.max_a):
            logging.error("a > max a in Voigt.")
        result[np.isnan(result)] = 0
        return result


def get_H(nu_norm, a):
    print("Re-computing Voigt profiles")

    H_values = []

    for nu_val in nu_norm:

        def integrand_H(y):
            return a / pi * np.exp(-(y**2)) / ((nu_val - y) ** 2 + a**2)

        integral = quad(integrand_H, -np.inf, np.inf)[0]
        H_values.append(integral)
    return np.array(H_values)


def get_L(nu_norm, a):
    print("Re-computing Voigt profiles")
    H_values = []

    for nu_val in nu_norm:

        def integrand_H(y):
            return (
                1 / pi * np.exp(-(y**2)) * (nu_val - y) / ((nu_val - y) ** 2 + a**2)
            )

        integral = quad(integrand_H, -np.inf, np.inf)[0]
        H_values.append(integral)
    return np.array(H_values)
