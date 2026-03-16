import logging

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_term_atom_model.object.atmosphere_parameters import (
    AtmosphereParameters,
)
from solrat.atom_model.shared.common_api.constant_property_slab import (
    ConstantPropertySlabAtmosphere,
)
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import (
    MultiSlabAtmosphere,
)
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter_IV_IpmV


def demo_constant_property_slab_multiline():
    """
    Demonstrate basic usage of ConstantPropertySlab for multiple non-overlapping line synthesis.
    """
    logging_config.init(logging.INFO)

    model_Mn = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
    model_Fe = PreconfiguredModels.multi_term_atom_lte_FeI_5434()
    model_Ni = PreconfiguredModels.multi_term_atom_lte_NiI_5435()

    reference_lambda_Mn = model_Mn.config.reference_lambda_A
    reference_lambda_Fe = model_Fe.config.reference_lambda_A
    reference_lambda_Ni = model_Ni.config.reference_lambda_A

    lambda_A_Mn = np.arange(reference_lambda_Mn + 1.5 - 0.5, reference_lambda_Mn + 1.5 + 0.5, 1e-3)
    lambda_A_Fe = np.arange(reference_lambda_Fe + 1.5 - 0.5, reference_lambda_Fe + 1.5 + 0.5, 1e-3)
    lambda_A_Ni = np.arange(reference_lambda_Ni + 1.5 - 0.5, reference_lambda_Ni + 1.5 + 0.5, 1e-3)

    nu_Mn = lambda_A_to_frequency_hz(lambda_A_Mn)
    nu_Fe = lambda_A_to_frequency_hz(lambda_A_Fe)
    nu_Ni = lambda_A_to_frequency_hz(lambda_A_Ni)

    radiation_tensor_Mn = model_Mn.RadiationTensor()
    radiation_tensor_Fe = model_Fe.RadiationTensor()
    radiation_tensor_Ni = model_Ni.RadiationTensor()

    # Test different magnetic field strengths
    plotter = StokesPlotter_IV_IpmV("Mn, Fe, and Ni lines:")

    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    # Atmosphere parameters:
    atmosphere1 = {
        "magnetic_field_gauss": 3000,
        "temperature_K": 4500,
        "macroscopic_velocity_cm_sm1": 0,
    }
    atmosphere2 = {
        "magnetic_field_gauss": 5000,
        "temperature_K": 5000,
        "macroscopic_velocity_cm_sm1": 0,
    }

    initial_stokes_Mn = Stokes.from_BP(nu_sm1=nu_Mn, temperature_K=5700)
    initial_stokes_Fe = Stokes.from_BP(nu_sm1=nu_Fe, temperature_K=5700)
    initial_stokes_Ni = Stokes.from_BP(nu_sm1=nu_Ni, temperature_K=5700)

    slab1_continuum_delta_tau = 0.05
    slab2_continuum_delta_tau = 0.005

    atmosphere_Mn = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model_Mn,
            radiation_tensor=radiation_tensor_Mn,
            line_delta_tau=1.6,
            continuum_delta_tau=slab1_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                model_config=model_Mn.config,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere1,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            model=model_Mn,
            radiation_tensor=radiation_tensor_Mn,
            line_delta_tau=0.01,
            continuum_delta_tau=slab2_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                model_config=model_Mn.config,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere2,
            ),
        ),
    )

    stokes_Mn = atmosphere_Mn.forward(initial_stokes=initial_stokes_Mn)

    atmosphere_Fe = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model_Fe,
            radiation_tensor=radiation_tensor_Fe,
            line_delta_tau=2.6,
            continuum_delta_tau=slab1_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                model_config=model_Fe.config,
                delta_v_turbulent_cm_sm1=2000_00,
                voigt_a=0,
                **atmosphere1,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            model=model_Fe,
            radiation_tensor=radiation_tensor_Fe,
            line_delta_tau=0.02,
            continuum_delta_tau=slab2_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                model_config=model_Fe.config,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere2,
            ),
        ),
    )

    stokes_Fe = atmosphere_Fe.forward(initial_stokes=initial_stokes_Fe)

    atmosphere_Ni = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model_Ni,
            radiation_tensor=radiation_tensor_Ni,
            line_delta_tau=1.5,
            continuum_delta_tau=slab1_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                model_config=model_Ni.config,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere1,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            model=model_Ni,
            radiation_tensor=radiation_tensor_Ni,
            line_delta_tau=0.01,
            continuum_delta_tau=slab2_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                model_config=model_Ni.config,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere2,
            ),
        ),
    )

    stokes_Ni = atmosphere_Ni.forward(initial_stokes=initial_stokes_Ni)

    plotter.add_stokes(
        lambda_A=np.concat([lambda_A_Mn, lambda_A_Fe, lambda_A_Ni]),
        reference_lambda_A=1.5,
        stokes=Stokes(
            nu=np.concat([stokes_Mn.nu, stokes_Fe.nu, stokes_Ni.nu]),
            I=np.concat([stokes_Mn.I, stokes_Fe.I, stokes_Ni.I]),
            Q=np.concat([stokes_Mn.Q, stokes_Fe.Q, stokes_Ni.Q]),
            U=np.concat([stokes_Mn.U, stokes_Fe.U, stokes_Ni.U]),
            V=np.concat([stokes_Mn.V, stokes_Fe.V, stokes_Ni.V]),
        ),
        stokes_reference=Stokes(
            nu=np.concat([initial_stokes_Mn.nu, initial_stokes_Fe.nu, initial_stokes_Ni.nu]),
            I=np.concat([initial_stokes_Mn.I, initial_stokes_Fe.I, initial_stokes_Ni.I]),
            Q=np.concat([initial_stokes_Mn.Q, initial_stokes_Fe.Q, initial_stokes_Ni.Q]),
            U=np.concat([initial_stokes_Mn.U, initial_stokes_Fe.U, initial_stokes_Ni.U]),
            V=np.concat([initial_stokes_Mn.V, initial_stokes_Fe.V, initial_stokes_Ni.V]),
        ),
        label="RTE with LTE SEE",
    )

    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_multiline()
