from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter_IV_IpmV


def main():
    r"""
    Zeeman effect in the Mn I 5432 A line (LTE) over a range of magnetic-field strengths.
    """
    setup_logging()

    model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.2,
        upper_wavelength_A=reference_lambda_A_air + 0.2,
        step_A=0.001,
    )

    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)

    plotter = StokesPlotter_IV_IpmV("Mn I 5432: Zeeman effect (LTE)", reference_lambda_A_air=reference_lambda_A_air)

    for B in [0, 500, 1000, 2000]:
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=B,
            temperature_K=5000,
            delta_v_turbulent_cm_sm1=100_000,
            macroscopic_velocity_cm_sm1=0,
            voigt_a=0,
        )
        atmosphere = MultiSlabAtmosphere(
            ConstantPropertySlabAtmosphere(
                model=model,
                radiation_tensor=model.RadiationTensor(),
                line_delta_tau=0.3,
                continuum_delta_tau=0.01,
                angles=angles,
                atmosphere_parameters=atmosphere_parameters,
            )
        )
        stokes = atmosphere.forward(initial_stokes=initial_stokes)

        plotter.add_stokes(
            nu=nu,
            stokes=stokes,
            norm=StokesPlotter_IV_IpmV.Norm.BY_REFERENCE,
            stokes_reference=initial_stokes,
            label=f"B = {B} G",
        )

    plotter.show()


if __name__ == "__main__":
    main()
