import logging
import unittest

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz


class TestConstantPropertySlab(unittest.TestCase):
    def test_constant_property_slab(self):
        """
        Sanity checks for ConstantPropertySlab with Mn I LTE synthesis.
        """
        logging_config.init(logging.INFO)

        model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
        reference_lambda = model.config.reference_lambda_A

        lambda_A = np.arange(reference_lambda - 0.5, reference_lambda + 0.5, 1e-3)
        nu = lambda_A_to_frequency_hz(lambda_A)

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

        assert np.all(np.isfinite(stokes_Mn.I))
        assert np.all(np.isfinite(stokes_Mn.V))
        # Absorption line: emergent I must dip below the background continuum
        assert np.min(stokes_Mn.I) < np.min(initial_stokes.I)
        # theta=0 (along B): no linear polarisation
        assert np.allclose(stokes_Mn.Q, 0, atol=1e-20)
        assert np.allclose(stokes_Mn.U, 0, atol=1e-20)
        # B=1000 G along LOS: circular polarisation must be present
        assert np.max(np.abs(stokes_Mn.V)) > 0
