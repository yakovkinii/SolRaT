import logging
import unittest

import numpy as np
from numpy import sqrt
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.utility.constants import kB_erg_Km1
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_erg


class TestStatisticalEquilibriumEquationsLTERealAtoms(unittest.TestCase):
    """
    Physics checks for the LTE SEE on real atomic data (Fe I 5434, Ni I 5435):
    trace normalisation, zero K!=0 coherences, correct Boltzmann ratios, and
    increasing excited fraction with temperature.
    """

    TEMPERATURE_K = 5000

    def _run_checks(self, model):
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=0,
            temperature_K=self.TEMPERATURE_K,
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config)

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        rho = see.get_solution()

        trace = 0.0
        for term in model.config.level_registry.terms.values():
            for level in term.levels:
                J = level.J
                trace += sqrt(2 * J + 1) * np.real(rho(term_id=term.term_id, K=0, Q=0, J=J, Jʹ=J))
        assert np.isclose(trace, 1.0, rtol=1e-10, atol=1e-10), f"Trace = {trace}, expected 1.0"

        for index, (term_id, K, Q, J, Jʹ) in see.matrix_builder.index_to_parameters.items():
            if K != 0:
                val = rho(term_id=term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                assert np.allclose(val, 0, rtol=1e-10, atol=1e-10), f"Expected 0 for K={K}, got {val}"

        T = self.TEMPERATURE_K
        for term in model.config.level_registry.terms.values():
            levels = sorted(term.levels, key=lambda lv: lv.energy_cmm1)
            for i in range(len(levels) - 1):
                lv_low = levels[i]
                lv_high = levels[i + 1]
                J_low, J_high = lv_low.J, lv_high.J
                delta_E_erg = energy_cmm1_to_erg(lv_high.energy_cmm1 - lv_low.energy_cmm1)

                rho_low = np.real(rho(term_id=term.term_id, K=0, Q=0, J=J_low, Jʹ=J_low))
                rho_high = np.real(rho(term_id=term.term_id, K=0, Q=0, J=J_high, Jʹ=J_high))

                if rho_low == 0:
                    continue

                actual_ratio = rho_high / rho_low
                expected_ratio = (sqrt(2 * J_high + 1) / sqrt(2 * J_low + 1)) * np.exp(-delta_E_erg / (kB_erg_Km1 * T))
                assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10, atol=1e-10), (
                    f"term={term.term_id} J={J_low} to {J_high}: "
                    f"ratio {actual_ratio:.8f} vs Boltzmann {expected_ratio:.8f}"
                )

        return rho

    def _excited_fraction(self, model, temperature_K):
        """Return fraction of population in excited terms."""
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=0,
            temperature_K=temperature_K,
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config)
        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        rho = see.get_solution()

        min_energy = min(lv.energy_cmm1 for term in model.config.level_registry.terms.values() for lv in term.levels)
        excited_pop = 0.0
        for term in model.config.level_registry.terms.values():
            for level in term.levels:
                if level.energy_cmm1 > min_energy:
                    excited_pop += np.real(rho(term_id=term.term_id, K=0, Q=0, J=level.J, Jʹ=level.J))
        return excited_pop

    def test_lte_FeI_5434(self):
        logging_config.init(logging.INFO)
        model = PreconfiguredModels.multi_term_atom_lte_FeI_5434()
        self._run_checks(model)

        frac_low = self._excited_fraction(model, temperature_K=3000)
        frac_high = self._excited_fraction(model, temperature_K=8000)
        assert frac_high > frac_low, "Excited fraction should increase with temperature"

    def test_lte_NiI_5435(self):
        logging_config.init(logging.INFO)
        model = PreconfiguredModels.multi_term_atom_lte_NiI_5435()
        self._run_checks(model)

        frac_low = self._excited_fraction(model, temperature_K=3000)
        frac_high = self._excited_fraction(model, temperature_K=8000)
        assert frac_high > frac_low, "Excited fraction should increase with temperature"
