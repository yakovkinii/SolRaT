import logging
import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_term_atom_model.data.HI import get_H_I_alpha_config
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.special import pseudo_hash


class TestHIAlphaSynthesis(unittest.TestCase):
    r"""
    A constant-property-slab synthesis of the H-alpha multi-term atom, reduced to a ``pseudo_hash``
    of the emergent Stokes vector and locked against a baseline.
    """

    def test_h_alpha_constant_slab_regression(self):
        setup_logging()
        model = Models.multi_term_atom().configure(config=get_H_I_alpha_config())
        reference_lambda_A_air = model.config.reference_lambda_A_air
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.3,
            upper_wavelength_A=reference_lambda_A_air + 0.3,
            step_A=5e-2,
        )

        angles = Angles(chi=0, theta=np.pi / 4, gamma=0, chi_B=0, theta_B=0)
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=30)

        atmosphere = MultiSlabAtmosphere(
            ConstantPropertySlabAtmosphere(
                model=model,
                radiation_tensor=radiation_tensor,
                line_delta_tau=0.1,
                continuum_delta_tau=0.01,
                angles=angles,
                atmosphere_parameters=model.AtmosphereParameters(
                    model_config=model.config,
                    magnetic_field_gauss=500.0,
                    temperature_K=8000.0,
                ),
            )
        )

        emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

        new_hash = pseudo_hash(emergent.I, emergent.Q, emergent.U, emergent.V)
        previous_hash = 4.938419942225e-06
        logging.info(f"test_h_alpha_constant_slab_regression current={new_hash!r} previous={previous_hash!r}")
        assert np.isclose(new_hash, previous_hash, rtol=1e-6, atol=0.0)


if __name__ == "__main__":
    unittest.main()
