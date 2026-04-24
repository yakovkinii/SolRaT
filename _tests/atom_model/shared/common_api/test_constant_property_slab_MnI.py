import logging
import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.special import pseudo_hash


class TestConstantPropertySlab(unittest.TestCase):
    def test_constant_property_slab(self):
        """
        Sanity checks for ConstantPropertySlab with Mn I LTE synthesis.
        """
        setup_logging()

        model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
        reference_lambda_A_air = model.config.reference_lambda_A_air
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.5,
            upper_wavelength_A=reference_lambda_A_air + 0.5,
            step_A=1e-3,
        )

        angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

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

        # Check that the result did not change from previous runs
        last_run_hash = 4.016455999176536e-05
        new_hash = pseudo_hash(stokes_Mn.I, stokes_Mn.Q, stokes_Mn.U, stokes_Mn.V)
        logging.info(new_hash)
        logging.info(last_run_hash)
        assert np.abs((last_run_hash - new_hash) / last_run_hash) < 1e-8
