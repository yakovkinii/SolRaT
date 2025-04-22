import numpy as np
from numpy import sqrt, exp, pi

from core.tensor.t_k_q import t_k_q
from core.utility.constant import kB, h, sqrt3, c, atomic_mass_unit, e_0, m_e
from core.utility.einstein_coefficients import (
    b_lu_from_b_ul_two_level_atom,
    b_ul_from_a_two_level_atom,
)
from core.utility.math import m1p
from core.utility.voigt import voigt
from core.utility.wigner_3j_6j_9j import wigner_6j


class TwoLevelAtom:
    def __init__(self, chi=0, theta=0, gamma=0):
        """
        theta, chi - direction of LOS wrt local coordinates
        gamma defines Q/U.
        """
        self.j_l = 2
        self.j_u = 3
        self.nu = (5579183 - 5069095) * 1e9  # Hz
        self.n = 1e1  # 1/cm3
        self.a_ul = 0.7e8  # 1/s
        self.temperature = 7000
        self.delta_nu_D = self.get_delta_nu_D()
        self.b_ul = b_ul_from_a_two_level_atom(self.a_ul, nu=self.nu)
        self.b_lu = b_lu_from_b_ul_two_level_atom(self.b_ul, j_u=self.j_u, j_l=self.j_l)
        self.rho_l = dict()
        for k in [0, 1, 2]:
            self.rho_l[k] = dict()

        self.rho_u = dict()
        for k in [0, 1, 2]:
            self.rho_u[k] = dict()

        self.t_k_q = dict()
        for k in [0, 1, 2]:
            self.t_k_q[k] = dict()
            for q in range(-k, k + 1):
                self.t_k_q[k][q] = dict()
                for i in [0, 1, 2, 3]:
                    self.t_k_q[k][q][i] = t_k_q(
                        k=k, q=q, i=i, chi=chi, theta=theta, gamma=gamma
                    )

    def get_delta_nu_D(self, atomic_mass=4, xi=0):
        return (
            sqrt(2 * kB * self.temperature / atomic_mass / atomic_mass_unit + xi**2)
            * self.nu
            / c
        )

    def _get_rho_u_k_q(self, k, q, j_k_q: dict):
        return (
                sqrt(3 * (2 * self.j_l + 1))
                * self.b_lu
                / self.a_ul
                * m1p(1 + self.j_l + self.j_u + q)
                * wigner_6j(1, 1, k, self.j_u, self.j_u, self.j_l)
                * j_k_q[k][-q]
        )

    def get_j_k_q_1d(self, stokes, nus):
        j_k_q = dict()
        for k in [0, 1, 2]:
            j_k_q[k] = dict()

        index = np.argmax(nus > self.nu)
        s_vector = stokes[index, :]
        for k in [0, 1, 2]:
            for q in range(-k, k + 1):
                sum_i = 0
                for i in [0, 1, 2, 3]:
                    sum_i += s_vector[i] * self.t_k_q[k][q][i]
                j_k_q[k][q] = sum_i
        return j_k_q

    def create_equations(self):
        """
        Create a set of equations to be solved for rho
        :return:
        """
        # 1. upper level.

    def _get_rho_u_k_q_lte(self, k, q):
        if k == 0 and q == 0:
            return exp(-(h * self.nu) / kB / self.temperature)
        else:
            return 0

    def calculate_rhos(self, j_k_q: dict = None, lte=False):
        for k in [1, 2]:
            for q in range(-k, k + 1):
                self.rho_l[k][q] = 0
        self.rho_l[0][0] = 1

        for k in [0, 1, 2]:
            for q in range(-k, k + 1):
                if lte:
                    self.rho_u[k][q] = self._get_rho_u_k_q_lte(k, q)
                else:
                    self.rho_u[k][q] = self._get_rho_u_k_q(k, q, j_k_q)

        scale_factor = sqrt(2 * self.j_u + 1) * self.rho_u[0][0] + sqrt(
            2 * self.j_l + 1
        )

        for k in [0, 1, 2]:
            for q in range(-k, k + 1):
                self.rho_u[k][q] /= scale_factor
                self.rho_l[k][q] /= scale_factor

    @staticmethod
    def get_Gamma(nu):
        gamma = 8 * pi**2 / 3 * e_0**2 / m_e / c**3 * nu**2
        return gamma / 3 / pi

    def get_k_matrix(self, nu):
        """
        7.16
        nu: vector l_nu
        nu is REGULAR frequency, not reduced
        """
        voigt_complex = voigt(
            nu=(self.nu - nu) / self.delta_nu_D, a=self.get_Gamma(nu) / self.delta_nu_D
        )
        # voigt_phi = np.real(voigt(nu=(self.nu - nu) / self.delta_nu_D, a=self.get_Gamma(nu) / self.delta_nu_D))
        # voigt_psi = np.imag(voigt(nu=(self.nu - nu) / self.delta_nu_D, a=self.get_Gamma(nu) / self.delta_nu_D))
        # import matplotlib.pyplot as plt; plt.plot(); plt.show()
        k_matrix = np.empty(
            (len(nu), 5, 4)
        )  # l_nu x [eta_a, rho_a, eta_s, rho_s, eps] x [I, Q, U, V]

        for i in [0, 1, 2, 3]:
            sum_k_q = 0
            for k in [0, 1, 2]:
                for q in range(-k, k + 1):
                    aaa = (
                        (
                                sqrt3
                                * m1p(1 + self.j_l + self.j_u + k)
                                * wigner_6j(1, 1, k, self.j_l, self.j_l, self.j_u)
                        )
                        * self.t_k_q[k][q][i]
                        * self.rho_l[k][q]
                    )
                    sum_k_q += aaa
            k_matrix[:, 0, i] = (
                h
                * nu
                / 4
                / pi
                * self.n
                * (2 * self.j_l + 1)
                * self.b_lu
                * np.real(sum_k_q * voigt_complex)
            )  # eta a
            k_matrix[:, 1, i] = (
                h
                * nu
                / 4
                / pi
                * self.n
                * (2 * self.j_l + 1)
                * self.b_lu
                * np.imag(sum_k_q * voigt_complex)
            )  # rho a

            sum_k_q = 0
            for k in [0, 1, 2]:
                for q in range(-k, k + 1):
                    aaa = (
                        (
                                sqrt3
                                * m1p(1 + self.j_l + self.j_u)
                                * wigner_6j(1, 1, k, self.j_u, self.j_u, self.j_l)
                        )
                        * self.t_k_q[k][q][i]
                        * self.rho_u[k][q]
                    )
                    sum_k_q += aaa
            k_matrix[:, 2, i] = (
                h
                * nu
                / 4
                / pi
                * self.n
                * (2 * self.j_u + 1)
                * self.b_ul
                * np.real(sum_k_q * voigt_complex)
            )  # eta s
            k_matrix[:, 3, i] = (
                h
                * nu
                / 4
                / pi
                * self.n
                * (2 * self.j_u + 1)
                * self.b_ul
                * np.imag(sum_k_q * voigt_complex)
            )  # rho s
            k_matrix[:, 4, i] = 2 * h * nu**3 / c**2 * k_matrix[:, 2, i]  # eps
        return k_matrix


# atom = TwoLevelAtom()
#
# atom.calculate_rhos(lte=True)
#
# nus = np.linspace(1e10-10,1e10+10,100)
# res=atom.get_k_matrix(nus)
