import logging
import time

import numpy as np
from matplotlib import pyplot as plt
from numpy import imag, real
from yatools import logging_config

from src.core.physics.functions import get_planck_BP
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


def main():
    """
    This demo shows the calculation of the eta_A, rho_A profiles.
    """
    logging_config.init(logging.INFO)

    term_registry = TermRegistry()
    term_registry.register_term(
        beta="1s",
        L=0,
        S=0.5,
        J=0.5,
        energy_cmm1=200_000,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=220_000,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=220_001,
    )
    term_registry.validate()

    nu = np.arange(5.995e14, 5.997e14, 1e8)  # Hz

    transition_registry = TransitionRegistry()
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="2p", L=1, S=0.5),
        level_lower=term_registry.get_level(beta="1s", L=0, S=0.5),
        einstein_a_ul_sm1=0.7e8,
    )

    atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=5_000_00)
    radiation_tensor = RadiationTensor(transition_registry=transition_registry)
    I0 = get_planck_BP(nu_sm1=nu, T_K=5000)
    radiation_tensor.fill_NLTE_near_isotropic(I0)

    atom = TwoTermAtom(
        term_registry=term_registry,
        transition_registry=transition_registry,
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor=radiation_tensor,
        disable_r_s=True,
        disable_n=True,
        n_frequencies=len(nu),
    )

    atom.add_all_equations()
    rho = atom.get_solution_direct()

    radiative_transfer_coefficients = RadiativeTransferCoefficients(
        atmosphere_parameters=atmosphere_parameters,
        transition_registry=transition_registry,
        nu=nu,
        theta=np.pi / 8,
        gamma=np.pi / 8,
        chi=np.pi / 8,
    )

    t0 = time.perf_counter()
    eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
    t1 = time.perf_counter()
    rho_sI = radiative_transfer_coefficients.rho_s(rho=rho, stokes_component_index=0)
    t2 = time.perf_counter()
    eta_rho_sI = radiative_transfer_coefficients.eta_rho_s(rho=rho, stokes_component_index=0)
    t3 = time.perf_counter()

    logging.info(f"eta_s: {t1 - t0:.2f} s")
    logging.info(f"rho_s: {t2 - t1:.2f} s")
    logging.info(f"eta_rho_s: {t3 - t2:.2f} s")

    plt.plot(nu, eta_sI, "g-", label=r"$\eta_s^I$")
    plt.plot(nu, rho_sI, "r-", label=r"$\rho_s^I$")
    plt.plot(nu, real(eta_rho_sI), "k:", label=r"$\eta_{\rho_s}^I$ (real)")
    plt.plot(nu, imag(eta_rho_sI), "b:", label=r"$\eta_{\rho_s}^I$ (imaginary)")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$\eta_s$")
    plt.title(r"$\eta_s$ vs Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
