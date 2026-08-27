import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz

from solrat.atom_model.shared.utility.constants import c_cm_sm1, sqrt_pi
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.voigt_profile import voigt


def main():
    r"""
    Absorption and dispersion (Faraday-Voigt) profiles of a normal Zeeman triplet (LL04 eq. 5.43,
    Fig. 5.3), checked against scipy.special.wofz.
    """
    setup_logging()

    nu0 = 5.996e14  # Hz

    wT = 100_00  # cm/s, thermal velocity
    delta_nu_D = nu0 * wT / c_cm_sm1
    Gamma = 0.05 * delta_nu_D

    nu_range = 12 * delta_nu_D
    nu = np.arange(nu0 - nu_range, nu0 + nu_range, delta_nu_D * 1e-2)  # Hz

    a = Gamma / delta_nu_D
    nuL = 5 * Gamma * 12  # for Fig 5.3
    nu_round_B = nuL / delta_nu_D
    nu_round = (nu0 - nu) / delta_nu_D
    wA = 0  # Macroscopic velocity, cm/s
    nu_round_A = wA / wT

    for alpha in [-1, 0, 1]:
        complex_voigt = voigt(nu=nu_round - nu_round_A + alpha * nu_round_B, a=a) / sqrt_pi / delta_nu_D
        phi = np.real(complex_voigt)
        psi = np.imag(complex_voigt)
        plt.plot((nu - nu0) / delta_nu_D, phi * delta_nu_D, label=rf"$\phi_{{{alpha}}}$")
        plt.plot((nu - nu0) / delta_nu_D, psi * delta_nu_D, ":", label=rf"$\psi_{{{alpha}}}$")

    plt.xlabel(r"$(\nu - \nu_0)/\Delta\nu_D$")
    plt.ylabel(r"$\phi$, $\psi$ (in units of $\Delta\nu_D$)")
    plt.title("Absorption and dispersion profiles")
    plt.xlim(-12, 12)
    plt.ylim(-0.6, 0.6)
    plt.legend()
    plt.grid()
    # Numeric check of the absorption profile H(a, v) = Re[w(v + i a)] against the reference Faddeeva
    # function (scipy.special.wofz); convention-independent, so it validates SolRaT's Voigt directly.
    v_grid = np.linspace(-12.0, 12.0, 2000)
    profile_residual = np.real(voigt(nu=v_grid, a=a)) - np.real(wofz(v_grid + 1j * a))
    profile_rms = float(np.sqrt(np.mean(profile_residual**2)))
    print(f"Voigt profile H(a, v) vs scipy.special.wofz: RMS SolRaT - reference = {profile_rms:.2e} (a = {a:.3f})")
    plt.show()


if __name__ == "__main__":
    main()
