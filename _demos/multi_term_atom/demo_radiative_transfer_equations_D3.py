from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter_IV


def main():
    r"""
    Stimulated-emission eta_S profiles for the He I D3 transition under super-strong magnetic fields,
    related to Fig. 8 of Yakovkin & Lozitsky (MNRAS, 2023, https://doi.org/10.1093/mnras/stad1816).
    """

    setup_logging()

    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 2,
        upper_wavelength_A=reference_lambda_A_air + 2,
        step_A=5e-4,
    )

    angles = Angles(
        chi=0,
        theta=0,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )

    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(
        h_arcsec=10,
    )

    plotter = StokesPlotter_IV(
        title="He I D3: Emission coefficient vs wavelength", reference_lambda_A_air=reference_lambda_A_air
    )

    for Bz in [20000, 40000, 60000, 80000, 100000]:
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=Bz,
            temperature_K=1000,
        )

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

        # Plot emission coefficient
        plotter.add(
            nu=nu,
            stokes_I=rtc.get_epsilon_I(),
            stokes_V=rtc.get_epsilon_V(),
            color="auto",
            label=rf"$B_z = {Bz/1000:.0f}$ kG",
        )

    plotter.show()


if __name__ == "__main__":
    main()
