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


def demo_constant_property_slab_MnI():
    """
    Demonstrate basic usage of ConstantPropertySlab for Mn I LTE line synthesis.
    """
    logging_config.init(logging.INFO)

    model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
    reference_lambda = model.config.reference_lambda_A

    lambda_A = np.arange(reference_lambda - 0.5, reference_lambda + 0.5, 1e-3)
    nu = lambda_A_to_frequency_hz(lambda_A)

    # Test different magnetic field strengths
    plotter = StokesPlotter_IV_IpmV("Mn I 5432 Line")

    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    # Atmosphere parameters:
    atmosphere1 = {
        "magnetic_field_gauss": 1000,
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
        lambda_A=np.concat([lambda_A]),
        lambda_ref_A=reference_lambda,
        norm=StokesNorm.BY_REFERENCE,
        stokes=Stokes(
            nu=np.concat([stokes_Mn.nu]),
            I=np.concat([stokes_Mn.I]),
            Q=np.concat([stokes_Mn.Q]),
            U=np.concat([stokes_Mn.U]),
            V=np.concat([stokes_Mn.V]),
        ),
        stokes_reference=Stokes(
            nu=np.concat([initial_stokes.nu]),
            I=np.concat([initial_stokes.I]),
            Q=np.concat([initial_stokes.Q]),
            U=np.concat([initial_stokes.U]),
            V=np.concat([initial_stokes.V]),
        ),
        label="RTE with LTE SEE",
    )

    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_MnI()
