import logging

import numpy as np

# import matplotlib.pyplot as plt
from numpy import cos, sin, pi, exp, sqrt, log10
import matplotlib.gridspec as gridspec

from yatools import logging_config

from core.utility.constant import c, h, kB
from pipeline.two_level_atom_resonance_polarization.atom import TwoLevelAtom

logging_config.init(logging.INFO)


# fig = plt.figure(figsize=(10, 10))
# gs = gridspec.GridSpec(3, 2)
# axTau = fig.add_subplot(gs[0, :])
# axI = fig.add_subplot(gs[1, 0])
# axQ = fig.add_subplot(gs[1, 1])
# axU = fig.add_subplot(gs[2, 0])
# axV = fig.add_subplot(gs[2, 1])
# axTau.set_ylabel("Atmospheric parameters")
# axTau.set_xlabel("log10(tau_c) [local continuum]")
# axTau.tick_params(axis="x", labelrotation=20)
# axI.set_ylabel("I / Ic")
# axI.set_xlabel("lambda (A)")
# axI.tick_params(axis="x", labelrotation=20)
# axQ.set_ylabel("Q / Ic")
# axQ.set_xlabel("lambda (A)")
# axQ.tick_params(axis="x", labelrotation=20)
# axU.set_ylabel("U / Ic")
# axU.set_xlabel("lambda (A)")
# axU.tick_params(axis="x", labelrotation=20)
# axV.set_ylabel("V / Ic")
# axV.set_xlabel("lambda (A)")
# axV.tick_params(axis="x", labelrotation=20)


def step(stokes, nu, k_matrix, temperature, dtau, kappa_l):
    """
    stokes: [n_nu, 4]
    k_matrix: l_nu x [eta_a, rho_a, eta_s, rho_s, eps] x [I, Q, U, V]

    """
    source_continuum = np.zeros_like(stokes)
    source_continuum[:, 0] = get_BP(nu, temperature)

    source = k_matrix[:, 4, :]
    k_square_matrix = np.empty((len(nu), 4, 4))
    # eta
    k_square_matrix[:, 0, :] = k_matrix[:, 0, :] + k_matrix[:, 2, :]
    k_square_matrix[:, 1:, 0] = k_square_matrix[:, 0, 1:]
    k_square_matrix[:, 1, 1] = k_square_matrix[:, 0, 0]
    k_square_matrix[:, 2, 2] = k_square_matrix[:, 0, 0]
    k_square_matrix[:, 3, 3] = k_square_matrix[:, 0, 0]
    # rho
    k_square_matrix[:, 1, 2] = k_matrix[:, 1, 3] + k_matrix[:, 3, 3]
    k_square_matrix[:, 2, 1] = -k_square_matrix[:, 1, 2]
    k_square_matrix[:, 1, 3] = -k_matrix[:, 1, 2] - k_matrix[:, 3, 2]
    k_square_matrix[:, 3, 1] = -k_square_matrix[:, 1, 3]
    k_square_matrix[:, 2, 3] = k_matrix[:, 1, 1] + k_matrix[:, 3, 1]
    k_square_matrix[:, 3, 2] = -k_square_matrix[:, 2, 3]

    dstokes = dtau * (
        stokes
        - source_continuum
        + kappa_l * np.einsum("ijk,ik->ij", k_square_matrix, stokes - source)
    )

    return stokes + dstokes


def get_nu_from_lambda(lambda_angstrem):
    return c / (lambda_angstrem * 1e-8)


def get_lambda_from_nu(nu):
    return c / (nu * 1e-8)


def get_BP(nu, T):
    return 2 * h * nu**3 / c**2 / (exp(h * nu / (kB * T) - 1))


def main():
    atom = TwoLevelAtom(theta=pi / 2, gamma=0, chi=0)
    nus = np.linspace(atom.nu - 1e11, atom.nu + 1e11, 1000)

    lambdas = get_lambda_from_nu(nus)

    I0 = get_BP(nus, atom.temperature)
    norm = 1  # max(I0)
    stokes = np.zeros((len(nus), 4))
    stokes[:, 0] = I0

    stokes0 = stokes.copy()

    stokes = stokes * 0
    #
    # axI.plot(lambdas, stokes[:, 0] / norm, label="I0")
    # axQ.plot(lambdas, stokes[:, 1] / norm, label="Q0")
    # axU.plot(lambdas, stokes[:, 2] / norm, label="U0")
    # axV.plot(lambdas, stokes[:, 3] / norm, label="V0")

    tau_values = []
    b_values = []
    t_values = []
    kappa_l_values = []
    tau = np.array(0.1)
    d_tau_max = -0.0001
    kappa_l = 1000
    i = 1
    while tau > 0.0001:
        tau_values.append(log10(tau))
        b_values.append(0)
        t_values.append(atom.temperature)
        kappa_l_values.append(kappa_l)
        dtau_true = max(d_tau_max, -tau / 1.1)
        tau = tau + dtau_true

        # zigzag NLTE approach:
        j_k_q = atom.get_j_k_q_1d(stokes0, nus)
        atom.calculate_rhos(j_k_q=j_k_q, lte=False)
        k_mat = atom.get_k_matrix(nus)

        stokes = step(stokes, nus, k_mat, atom.temperature, dtau_true, kappa_l)
        # if i % 20 == 0:
        #     axI.plot(lambdas, stokes[:, 0] / norm, label=f"tau={tau.round(5)}")
        #     axQ.plot(lambdas, stokes[:, 1] / norm, label=f"tau={tau.round(5)}")
        #     axU.plot(lambdas, stokes[:, 2] / norm, label=f"tau={tau.round(5)}")
        #     axV.plot(lambdas, stokes[:, 3] / norm, label=f"tau={tau.round(5)}")
        i = i + 1

    # axTau.plot(tau_values, np.array(b_values) / 1000, label="B (kG)")
    # axTau.plot(tau_values, np.array(t_values) / 10000, label="T (*10000 K)")
    # axTau.plot(tau_values, kappa_l_values, label="kappa_L")
    # axI.plot(lambdas, stokes[:, 0] / norm, label="I")
    # axQ.plot(lambdas, stokes[:, 1] / norm, label="Q")
    # axU.plot(lambdas, stokes[:, 2] / norm, label="U")
    # axV.plot(lambdas, stokes[:, 3] / norm, label="V")
    # axTau.legend()
    # axI.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
