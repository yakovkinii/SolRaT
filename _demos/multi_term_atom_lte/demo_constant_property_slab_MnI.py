import logging

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter_IV_IpmV


def demo_constant_property_slab_MnI():
    """
    Demonstrate basic usage of ConstantPropertySlab for Mn I LTE line synthesis.
    """
    setup_logging(logging.INFO)

    model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.2,
        upper_wavelength_A=reference_lambda_A_air + 0.2,
        step_A=1e-3,
    )

    # Test different magnetic field strengths
    plotter = StokesPlotter_IV_IpmV("Mn I 5432 Line", reference_lambda_A_air=reference_lambda_A_air)

    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    # Atmosphere parameters:
    atmosphere1 = {
        "magnetic_field_gauss": 500,
        "temperature_K": 5000,
        "delta_v_turbulent_cm_sm1": 1000_00,
        "macroscopic_velocity_cm_sm1": 0,
        "voigt_a": 0,
    }

    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)

    atmosphere_Mn = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=model.RadiationTensor(),
            line_delta_tau=0.3,
            continuum_delta_tau=0.01,
            angles=angles,
            atmosphere_parameters=model.AtmosphereParameters(
                model_config=model.config,
                **atmosphere1,
            ),
        ),
    )

    stokes_Mn = atmosphere_Mn.forward(initial_stokes=initial_stokes)

    plotter.add_stokes(
        nu=nu,
        norm=StokesPlotter_IV_IpmV.Norm.BY_REFERENCE,
        stokes=stokes_Mn,
        stokes_reference=initial_stokes,
        label="RTE with LTE SEE",
    )

    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_MnI()
