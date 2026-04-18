import logging

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter


def main():
    """
    This demo shows the calculation of the stimulated emission eta_S profiles
    in the case of a two-term atom under anisotropic irradiation.
    The inclined POV results in non-zero Stokes I, Q, and U parameters
    even when no magnetic field is present (Stokes V is still 0 since J1Q is 0).
    The results are compared with the analytical solution.
    Reference: (LL04 10.127)
    """

    logging_config.init(logging.INFO)

    model = PreconfiguredModels.multi_term_atom_mock()
    reference_nu = lambda_A_to_frequency_hz(model.config.reference_lambda_A)

    nu = np.arange(reference_nu - 1e11, reference_nu + 1e11, 1e8)  # Hz

    angles = Angles(
        chi=0,
        theta=np.pi / 4,
        gamma=np.pi / 8,
        chi_B=0,
        theta_B=0,
    )

    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(
        h_arcsec=30,
    )

    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=0,
        temperature_K=5500,
        delta_v_turbulent_cm_sm1=50_00,
    )

    plotter = StokesPlotter(r"$\eta_s$ vs Frequency", x_label=r"$\nu$ (1/s)")

    # Construct SEE
    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
    )

    # Solve SEE
    rho = see.get_solution()

    # Solve RTE
    rtc = rte.calculate_all_coefficients(
        atmosphere_parameters=atmosphere_parameters,
        rho=rho,
        angles=angles,
    )

    # Analytical expressions:
    model_legacy = PreconfiguredModels.multi_term_atom_legacy_mock()
    rte_legacy = model_legacy.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

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

    plotter.add(
        lambda_A=nu,
        lambda_ref_A=0,
        stokes_I=np.real(rtc.eta_rho_sI) / np.max(np.abs(eta_sI_analytic)),
        stokes_Q=np.real(rtc.eta_rho_sQ) / np.max(np.abs(eta_sI_analytic)),
        stokes_U=np.real(rtc.eta_rho_sU) / np.max(np.abs(eta_sI_analytic)),
        stokes_V=np.real(rtc.eta_rho_sV) / np.max(np.abs(eta_sI_analytic)),
        label="SEE+RTE implementation",
    )
    plotter.add(
        lambda_A=nu,
        lambda_ref_A=0,
        stokes_I=eta_sI_analytic / np.max(np.abs(eta_sI_analytic)),
        stokes_Q=eta_sQ_analytic / np.max(np.abs(eta_sI_analytic)),
        stokes_U=eta_sU_analytic / np.max(np.abs(eta_sI_analytic)),
        stokes_V=eta_sV_analytic / np.max(np.abs(eta_sI_analytic)),
        label="Analytical solution",
        style="--",
        linewidth=2,
    )
    plotter.axs[3].set_ylim(-1, 1)
    plotter.show()


if __name__ == "__main__":
    main()
