import logging
import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


class TestStatisticalEquilibriumEquationsResonanceStimulated(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance_stimulated(self):
        setup_logging()

        model = PreconfiguredModels.multi_term_atom_mock()
        model_legacy = PreconfiguredModels.multi_term_atom_legacy_mock()

        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config, magnetic_field_gauss=0, temperature_K=7000
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(h_arcsec=30)

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see_legacy = model_legacy.StatisticalEquilibriumEquations.from_model_config(model_legacy.config)

        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        see_legacy.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )

        rho_legacy = see_legacy.get_solution()
        rho = see.get_solution()

        for term in model.config.level_registry.terms.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                logging.info(f"{term.term_id}, {K}, {Q}, {J}, {Jʹ}")
                logging.info(rho(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ))
                logging.info(rho_legacy(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ))
                assert (
                    np.abs(
                        rho(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                        - rho_legacy(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                    ).max()
                    < 1e-10
                )
