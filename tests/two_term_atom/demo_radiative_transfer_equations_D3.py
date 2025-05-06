import logging

import numpy as np
from matplotlib import pyplot as plt
from yatools import logging_config

from src.core.physics.functions import get_planck_BP, lambda_cm_to_frequency_hz
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


def main():
    """
    This demo shows the calculation of stimulated emission eta_S profiles
    for the He I D3 transition under super-strong magnetic field of 20 kG.
    This result is closely related to Fig. 8 in Yakovkin & Lozitsky (MNRAS, 2023) https://doi.org/10.1093/mnras/stad1816
    (in the latter, the Stokes profiles are shown instead, which should resemble eta_S for low optical depth).
    This demo takes ~ 50 seconds to run.
    """

    logging_config.init(logging.INFO)
    term_registry = TermRegistry()
    term_registry.register_term(
        beta="2s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=159855.9726,
    )
    term_registry.register_term(
        beta="3s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=183236.7905,
    )
    term_registry.register_term(
        beta="2p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=169087.8291,
    )
    term_registry.register_term(
        beta="2p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=169086.8412,
    )
    term_registry.register_term(
        beta="2p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=169086.7647,
    )

    term_registry.register_term(
        beta="3p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=185564.8528,
    )
    term_registry.register_term(
        beta="3p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=185564.5817,
    )
    term_registry.register_term(
        beta="3p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=185564.5602,
    )
    term_registry.register_term(
        beta="3d3",
        L=2,
        S=1,
        J=1,
        energy_cmm1=186101.5908,
    )
    term_registry.register_term(
        beta="3d3",
        L=2,
        S=1,
        J=2,
        energy_cmm1=186101.5466,
    )
    term_registry.register_term(
        beta="3d3",
        L=2,
        S=1,
        J=3,
        energy_cmm1=186101.5440,
    )
    term_registry.validate()

    lambda_A = np.arange(5876, 5879, 2e-3)
    lambda_cm = lambda_A * 1e-8  # nm
    nu = lambda_cm_to_frequency_hz(lambda_cm)  # Hz
    transition_registry = TransitionRegistry()

    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="2p3", L=1, S=1),
        level_lower=term_registry.get_level(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=3 * 1.022e7,
    )
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="3p3", L=1, S=1),
        level_lower=term_registry.get_level(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=3 * 9.478e6,
    )
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="3s3", L=0, S=1),
        level_lower=term_registry.get_level(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=3.080e6 + 9.259e6 + 1.540e7,
    )
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="3d3", L=2, S=1),
        level_lower=term_registry.get_level(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=3.920e7 + 5.290e7 + 2.940e7 + 7.060e7 + 1.760e7 + 1.960e6,
    )

    magnetic_field_gauss = 20000
    atmosphere_parameters = AtmosphereParameters(
        magnetic_field_gauss=magnetic_field_gauss, delta_v_thermal_cm_sm1=1_000_00
    )
    radiation_tensor = RadiationTensor(transition_registry=transition_registry)
    I0 = get_planck_BP(nu_sm1=nu, T_K=5000)
    radiation_tensor.fill_isotropic(I0)
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
    )
    eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
    eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)

    plt.plot(lambda_A - 5877.23, eta_sI / max(eta_sI), label=r"$\eta_s$ (Stokes $I$)")
    plt.plot(lambda_A - 5877.23, eta_sV / max(eta_sI), label=r"$\eta_s$ (Stokes $V$)")
    plt.xlabel(r"$\Delta\lambda$ ($\AA$)")
    plt.ylabel(r"$\eta_s$ (a.u.)")
    plt.title(rf"He I D3: $\eta_s$ vs $\Delta\lambda$. $B_z = {magnetic_field_gauss//1000}$ kG")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
