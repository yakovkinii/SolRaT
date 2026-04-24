import logging

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter_IV


def main():
    """
    This demo shows the calculation of stimulated emission eta_S profiles
    for the He I D3 transition under super-strong magnetic fields.
    This result is related to Fig. 8 in Yakovkin & Lozitsky (MNRAS, 2023)
    https://doi.org/10.1093/mnras/stad1816, where these profiles were obtained using HAZEL2.
    In the mentioned paper, the Stokes profiles are shown instead,
    but they match (after normalization) eta_S for low optical depths.
    """

    setup_logging(logging.INFO)

    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.2,
        upper_wavelength_A=reference_lambda_A_air + 0.5,
        step_A=2e-3,
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

    # Fill the radiation tensor with anisotropic radiation field 10 arcsec from the Sun's apparent surface
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(h_arcsec=10)

    # Set up the plotter
    plotter = StokesPlotter_IV(
        "He I D3 emission at different magnetic fields", reference_lambda_A_air=reference_lambda_A_air
    )

    # loop through the magnetic field values
    for Bz in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=Bz,
            temperature_K=1000,  # Low temperature to see the details of fine structure
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
            label=rf"$B_z = {Bz/1000:.1f}$ kG",
            linewidth=2,
        )

    plotter.show()


if __name__ == "__main__":
    main()
