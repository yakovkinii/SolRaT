import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.constants import h_erg_s, kB_erg_Km1, sqrt2
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestStatisticalEquilibriumEquationsLTE(unittest.TestCase):
    def test_statistical_equilibrium_equations_lte(self):
        # (10.126)
        setup_logging()

        model = PreconfiguredModels.multi_term_atom_mock_nofs()

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=0,
            temperature_K=10000,
        )
        radiation_tensor = (
            model.RadiationTensor.from_model_config(config=model.config)
            .fill_planck(temperature_K=atmosphere_parameters.temperature_K)
            .rotate_to_magnetic_frame(angles=angles)
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)

        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )

        model_lte = PreconfiguredModels.multi_term_atom_mock_nofs_lte()
        see_lte = model_lte.StatisticalEquilibriumEquations.from_model_config(model_lte.config)
        see_lte.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )

        rho = see.get_solution()
        rho_lte = see_lte.get_solution()

        exp_hnu_kT = np.exp(
            -h_erg_s
            * next(iter(model.config.transition_registry.transitions.values())).get_mean_transition_frequency_sm1()
            / kB_erg_Km1
            / atmosphere_parameters.temperature_K
        )
        rho_analytical = {
            "1s_L=0.0_S=0.5_K=0.0_Q=0.0_J=0.5_Jʹ=0.5": 1 / sqrt2 / (1 + 3 * exp_hnu_kT),
            "2p_L=1.0_S=0.5_K=0.0_Q=0.0_J=0.5_Jʹ=0.5": exp_hnu_kT / sqrt2 / (1 + 3 * exp_hnu_kT),
            "2p_L=1.0_S=0.5_K=0.0_Q=0.0_J=1.5_Jʹ=1.5": exp_hnu_kT / (1 + 3 * exp_hnu_kT),
        }

        for coherence_id, coherence in rho.data.items():
            coherence_lte = rho_lte.data[coherence_id]
            if coherence_lte == 0:
                assert np.abs(coherence) < 1e-15
            else:
                assert np.abs((coherence_lte - coherence) / coherence_lte) < 1e-15
                if coherence_id in rho_analytical:
                    coherence_analytical = rho_analytical[coherence_id]
                    assert np.abs((coherence_lte - coherence_analytical) / coherence_analytical) < 1e-15
                    assert np.abs((coherence - coherence_analytical) / coherence_analytical) < 1e-15
