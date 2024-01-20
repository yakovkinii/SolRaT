import logging
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, pi, exp, sqrt, log10
import matplotlib.gridspec as gridspec
import core.utility.voigt as v

# SGS
from voigt import Voigt

from yatools import logging_config

logging_config.init(logging.INFO)


fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 2)
axTau = fig.add_subplot(gs[0, :])  # Large subplot A (row 0, col 0)
axI = fig.add_subplot(gs[1, 0])  # Small subplot B (row 0, col 1)
axQ = fig.add_subplot(gs[1, 1])  # Small subplot C (row 1, col 1)
axU = fig.add_subplot(gs[2, 0])  # Small subplot C (row 1, col 1)
axV = fig.add_subplot(gs[2, 1])  # Small subplot C (row 1, col 1)
axTau.set_ylabel("Atmospheric parameters")
axTau.set_xlabel("log10(tau_c) [local continuum]")
axTau.tick_params(axis="x", labelrotation=20)
axI.set_ylabel("I / Ic")
axI.set_xlabel("lambda (A)")
axI.tick_params(axis="x", labelrotation=20)
axQ.set_ylabel("Q / Ic")
axQ.set_xlabel("lambda (A)")
axQ.tick_params(axis="x", labelrotation=20)
axU.set_ylabel("U / Ic")
axU.set_xlabel("lambda (A)")
axU.tick_params(axis="x", labelrotation=20)
axV.set_ylabel("V / Ic")
axV.set_xlabel("lambda (A)")
axV.tick_params(axis="x", labelrotation=20)


def get_k_matrix(theta, chi, delta_nu_D, Gamma, nu, nu_0, B):
    nu_L = mu0 * B / h

    phi_b = (
        1
        / sqrt(pi)
        / delta_nu_D
        * np.real(v.voigt(nu=(nu_0 + nu_L - nu) / delta_nu_D, a=Gamma / delta_nu_D))
    )
    phi_p = (
        1
        / sqrt(pi)
        / delta_nu_D
        * np.real(v.voigt(nu=(nu_0 - nu) / delta_nu_D, a=Gamma / delta_nu_D))
    )
    phi_r = (
        1
        / sqrt(pi)
        / delta_nu_D
        * np.real(v.voigt(nu=(nu_0 - nu_L - nu) / delta_nu_D, a=Gamma / delta_nu_D))
    )
    psi_b = (
        1
        / sqrt(pi)
        / delta_nu_D
        * np.imag(v.voigt(nu=(nu_0 + nu_L - nu) / delta_nu_D, a=Gamma / delta_nu_D))
    )
    psi_p = (
        1
        / sqrt(pi)
        / delta_nu_D
        * np.imag(v.voigt(nu=(nu_0 - nu) / delta_nu_D, a=Gamma / delta_nu_D))
    )
    psi_r = (
        1
        / sqrt(pi)
        / delta_nu_D
        * np.imag(v.voigt(nu=(nu_0 - nu_L - nu) / delta_nu_D, a=Gamma / delta_nu_D))
    )

    eta_p = delta_nu_D * phi_p
    eta_r = delta_nu_D * phi_r
    eta_b = delta_nu_D * phi_b
    rho_p = delta_nu_D * psi_p
    rho_r = delta_nu_D * psi_r
    rho_b = delta_nu_D * psi_b

    h_I = 0.5 * (
        eta_p * sin(theta) ** 2 + 0.5 * (eta_b + eta_r) * (1 + cos(theta) ** 2)
    )
    h_Q = 0.5 * (eta_p - 0.5 * (eta_b + eta_r)) * sin(theta) ** 2 * cos(2 * chi)
    h_U = 0.5 * (eta_p - 0.5 * (eta_b + eta_r)) * sin(theta) ** 2 * sin(2 * chi)
    h_V = 0.5 * (eta_r - eta_b) * cos(theta)
    r_Q = 0.5 * (rho_p - (rho_b + rho_r) / 2) * sin(theta) ** 2 * cos(2 * chi)
    r_U = 0.5 * (rho_p - (rho_b + rho_r) / 2) * sin(theta) ** 2 * sin(2 * chi)
    r_V = 0.5 * (rho_r - rho_b) * cos(theta)
    return [h_I, h_Q, h_U, h_V, r_Q, r_U, r_V]


def get_BP(nu, T):
    return 2 * h * nu**3 / c**2 / (exp(h * nu / (kB * T) - 1))


def get_theta(tau_c):
    return pi / 2


def get_chi(tau_c):
    return pi / 2


def get_T(tau_c):
    return 5000 + 100000 * exp(-(tau_c**2) / 0.01)


def get_delta_nu_D(tau_c, mu=4, xi=0):
    """
    :param tau_c:
    :param mu: atomic mass number
    :param xi: turbulent velocity cm/s
    :return:
    """
    return sqrt(2 * kB * get_T(tau_c) / mu / M + xi**2) * nu_0 / c


def get_Gamma(nu):
    gamma = 8 * pi**2 / 3 * e_0**2 / m_e / c**3 * nu**2
    return gamma / 3 / pi


def get_B(tau_c):
    return 30000 * exp(-((tau_c - 0.1) ** 2) / 0.01)


def get_kappa_L(tau_c):
    return 10 * exp(-((tau_c - 0.01) ** 2) / 0.001)


def step(I, Q, U, V, nu, tau_c, dTau):
    theta = get_theta(tau_c)
    chi = get_chi(tau_c)
    delta_nu_D = get_delta_nu_D(tau_c)
    Gamma = get_Gamma(nu)
    B = get_B(tau_c)
    kappa_L = get_kappa_L(tau_c)
    T = get_T(tau_c)

    h_I, h_Q, h_U, h_V, r_Q, r_U, r_V = get_k_matrix(
        theta, chi, delta_nu_D, Gamma, nu, nu_0, B
    )
    SC = get_BP(nu, T)
    SL = get_BP(nu, T)

    I1 = (
        I
        + dTau * (I - SC)
        + dTau * kappa_L * (h_I * (I - SL) + h_Q * Q + h_U * U + h_V * V)
    )

    Q1 = Q + dTau * Q + dTau * kappa_L * (h_Q * (I - SL) + h_I * Q + r_V * U - r_U * V)

    U1 = U + dTau * kappa_L * (h_U * (I - SL) - r_V * Q + h_I * U + r_Q * V)

    V1 = V + dTau * kappa_L * (h_V * (I - SL) + r_U * Q - r_Q * U + h_I * V)
    return [I1, Q1, U1, V1]


def get_nu_from_lambda(lambda_angstrem):
    return c / (lambda_angstrem * 1e-8)


def main():
    lambdas = np.array(np.arange(4999, 5001, 0.01))
    nu = get_nu_from_lambda(lambdas)

    I0 = get_BP(nu, 7000)

    I = I0
    Q = np.zeros_like(nu)
    U = np.zeros_like(nu)
    V = np.zeros_like(nu)
    # ax[0,0].plot(lambdas, I, label="I0")
    # ax[0,1].plot(lambdas, Q, label="Q0")
    # ax[1,0].plot(lambdas, U, label="U0")
    # ax[1,1].plot(lambdas, V, label="V0")
    tau_values = []
    B_values = []
    T_values = []
    kappa_L_values = []
    tau = np.array(2.0)
    dTau_max = -0.05
    i = 1
    while tau > 0.0001:
        tau_values.append(log10(tau))
        B_values.append(get_B(tau))
        T_values.append(get_T(tau))
        kappa_L_values.append(get_kappa_L(tau))
        dtau_true = max(dTau_max, -tau / 2)
        tau = tau + dtau_true
        I, Q, U, V = step(I, Q, U, V, nu, tau, dtau_true)
        # if i%50==0:
        #     ax[0, 0].plot(lambdas, I, label=f"tau={tau.round(5)}")
        #     ax[0, 1].plot(lambdas, Q, label=f"tau={tau.round(5)}")
        #     ax[1, 0].plot(lambdas, U, label=f"tau={tau.round(5)}")
        #     ax[1, 1].plot(lambdas, V, label=f"tau={tau.round(5)}")
        i = i + 1
    axTau.plot(tau_values, np.array(B_values) / 1000, label="B (kG)")
    axTau.plot(tau_values, np.array(T_values) / 10000, label="T (*10000 K)")
    axTau.plot(tau_values, kappa_L_values, label="kappa_L")
    axI.plot(lambdas, I / I[0])
    axQ.plot(lambdas, Q / I[0])
    axU.plot(lambdas, U / I[0])
    axV.plot(lambdas, V / I[0])
    axTau.legend()
    # mng = plt.get_current_fig_manager()

    # Maximize the figure window
    # mng.window.state('zoomed')  # For TkAgg backend
    # mng.resize(*mng.window.maxsize())  # Alternative for some other backends
    plt.tight_layout()

    plt.show()


nu_0 = get_nu_from_lambda(5000)
main()
