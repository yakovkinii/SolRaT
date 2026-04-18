import logging

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesNorm, StokesPlotter_IV_IpmV


def main():
    """
    Zeeman effect in the Mn I 5432 A line (LTE) for a range of magnetic field strengths.
    """
    logging_config.init(logging.INFO)

    model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()

    reference_lambda = model.config.reference_lambda_A
    lambda_A = np.arange(reference_lambda - 0.6, reference_lambda + 0.6, 0.005)
    nu = lambda_A_to_frequency_hz(lambda_A)

    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)

    plotter = StokesPlotter_IV_IpmV("Mn I 5432 — Zeeman effect (LTE)")

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
            lambda_A=lambda_A,
            lambda_ref_A=reference_lambda,
            stokes=stokes,
            norm=StokesNorm.BY_REFERENCE,
            stokes_reference=initial_stokes,
            label=f"B = {B} G",
        )

    plotter.show()


if __name__ == "__main__":
    main()
