import logging

import numpy as np
from yatools import logging_config

from src.gui.plots.plot_stokes_profiles import StokesPlotter
from src.multi_term_atom.atomic_data.mock import get_mock_atom_data
from src.multi_term_atom.legacy.radiative_transfer_equations_legacy import (
    MultiTermAtomRTELegacy,
)
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


def main():
    """
    This demo shows the calculation of the stimulated emission eta_S profiles
    in the case of a two-term atom under anisotropic irradiation.
    The inclined POV results in non-zero Stokes I, Q, U, and V parameters
    even when no magnetic field is present.
    The results are compared with the analytical solution.
    Reference: (10.127)
    """
    logging_config.init(logging.INFO)

    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu = get_mock_atom_data()
    nu = np.arange(reference_nu_sm1 - 1e11, reference_nu_sm1 + 1e11, 1e8)  # Hz

    atmosphere_parameters = AtmosphereParameters(
        magnetic_field_gauss=0, atomic_mass_amu=atomic_mass_amu, temperature_K=5500, delta_v_turbulent_cm_sm1=50_00
    )
    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=30)

    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        disable_r_s=True,
    )

    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=radiation_tensor,
    )
    rho = see.get_solution()

    rte_legacy = MultiTermAtomRTELegacy(
        transition_registry=transition_registry,
        nu=nu,
    )
    rte = MultiTermAtomRTE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        nu=nu,
    )
    angles = Angles(
        chi=np.pi / 7,
        theta=np.pi / 7,
        gamma=np.pi / 7,
        chi_B=np.pi / 5,
        theta_B=np.pi / 5,
    )
    rtc = rte.calculate_all_coefficients(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
    eta_sI, eta_sQ, eta_sU, eta_sV = rtc.eta_rho_sI, rtc.eta_rho_sQ, rtc.eta_rho_sU, rtc.eta_rho_sV
    eta_sI_analytic = rte_legacy.eta_s_no_field(
        rho=rho, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
    )
    eta_sQ_analytic = rte_legacy.eta_s_no_field(
        rho=rho, stokes_component_index=1, atmosphere_parameters=atmosphere_parameters, angles=angles
    )
    eta_sU_analytic = rte_legacy.eta_s_no_field(
        rho=rho, stokes_component_index=2, atmosphere_parameters=atmosphere_parameters, angles=angles
    )
    eta_sV_analytic = rte_legacy.eta_s_no_field(
        rho=rho, stokes_component_index=3, atmosphere_parameters=atmosphere_parameters, angles=angles
    )

    plotter = StokesPlotter(r"$\eta_s$ vs Frequency", x_label="$\nu$ (1/s)")
    plotter.add(
        nu,
        0,
        np.real(eta_sI) / np.max(np.abs(eta_sI_analytic)),
        np.real(eta_sQ) / np.max(np.abs(eta_sI_analytic)),
        np.real(eta_sU) / np.max(np.abs(eta_sI_analytic)),
        np.real(eta_sV) / np.max(np.abs(eta_sI_analytic)),
        label="SEE+RTE implementation",
    )
    plotter.add(
        nu,
        0,
        eta_sI_analytic / np.max(np.abs(eta_sI_analytic)),
        eta_sQ_analytic / np.max(np.abs(eta_sI_analytic)),
        eta_sU_analytic / np.max(np.abs(eta_sI_analytic)),
        eta_sV_analytic / np.max(np.abs(eta_sI_analytic)),
        label="Analytical solution",
        style="--",
        linewidth=2,
    )
    # Todo investigate sV
    plotter.show()


if __name__ == "__main__":
    main()
