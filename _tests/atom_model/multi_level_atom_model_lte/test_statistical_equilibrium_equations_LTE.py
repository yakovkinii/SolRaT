import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.constants import h_erg_s, kB_erg_Km1
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestStatisticalEquilibriumEquationsLTE(unittest.TestCase):
    r"""
    Known-limit check on the multi-level LTE SEE. Illuminated by an isotropic Planck field at B = 0 the
    full (NLTE) multi-level SEE relaxes to detailed balance, so its solution must coincide with the LTE
    Boltzmann solution and with the analytic populations :math:`\rho^0_0(\alpha J)\propto\sqrt{2J+1}\,
    e^{-E_J/kT}` of the J = 0 -> 1 mock atom.
    """

    def test_isotropic_planck_reduces_to_lte_boltzmann(self):
        setup_logging()

        temperature_K = 10000.0
        angles = Angles(chi=np.pi / 5, theta=np.pi / 7, gamma=np.pi / 9, chi_B=np.pi / 3, theta_B=np.pi / 5)

        model = PreconfiguredModels.multi_level_atom_mock()
        model_lte = PreconfiguredModels.multi_level_atom_mock_lte()

        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
        )
        radiation_tensor = (
            model.RadiationTensor.from_model_config(config=model.config)
            .fill_planck(temperature_K=temperature_K)
            .rotate_to_magnetic_frame(angles=angles)
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see.fill_all_equations(atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor)
        rho = see.get_solution()

        see_lte = model_lte.StatisticalEquilibriumEquations.from_model_config(model_lte.config)
        see_lte.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
        )
        rho_lte = see_lte.get_solution()

        transition = next(iter(model.config.transition_registry.transitions.values()))
        lower_id = transition.level_lower.level_id
        upper_id = transition.level_upper.level_id
        exp_hnu_kT = np.exp(-h_erg_s * transition.get_mean_transition_frequency_sm1() / kB_erg_Km1 / temperature_K)
        trace = 1.0 + 3.0 * exp_hnu_kT
        rho00_lower_analytic = 1.0 / trace
        rho00_upper_analytic = np.sqrt(3.0) * exp_hnu_kT / trace

        # The isotropic-Planck NLTE solution equals the LTE solution coherence by coherence.
        for coherence_id, value_lte in rho_lte.data.items():
            self.assertLess(np.abs(rho.data[coherence_id] - value_lte), 1e-10 + 1e-8 * np.abs(value_lte))

        # Both reproduce the analytic Boltzmann populations.
        self.assertLess(np.abs(rho_lte(0, 0, lower_id) - rho00_lower_analytic), 1e-12)
        self.assertLess(np.abs(rho_lte(0, 0, upper_id) - rho00_upper_analytic), 1e-12)
        self.assertLess(np.abs(rho(0, 0, lower_id) - rho00_lower_analytic), 1e-9)
        self.assertLess(np.abs(rho(0, 0, upper_id) - rho00_upper_analytic), 1e-9)

        # An isotropic field creates no upper-level alignment in either solution.
        self.assertLess(np.abs(rho_lte(2, 0, upper_id)), 1e-14)
        self.assertLess(np.abs(rho(2, 0, upper_id)), 1e-9)


if __name__ == "__main__":
    unittest.main()
