import logging
import unittest

import numpy as np
from yatools import logging_config

from src.core.engine.functions.looping import TRIANGULAR, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.two_term_atom.atomic_data.mock import get_mock_atom_data
from src.two_term_atom.legacy.statistical_equilibrium_equations_legacy import TwoTermAtomSEELegacy
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtomSEE


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        """
        Test that SEE are runnable with stimulated emission enabled
        """
        logging_config.init(logging.INFO)

        term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=500_00)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(
            h_arcsec=30
        )
        see_legacy = TwoTermAtomSEELegacy(
            term_registry=term_registry,
            transition_registry=transition_registry,
            disable_r_s=False,
            disable_n=False,
        )

        see = TwoTermAtomSEE(
            term_registry=term_registry,
            transition_registry=transition_registry,
            disable_r_s=False,
            disable_n=False,
        )

        see_legacy.add_all_equations(atmosphere_parameters=atmosphere_parameters, radiation_tensor=radiation_tensor)
        see.add_all_equations(atmosphere_parameters=atmosphere_parameters, radiation_tensor=radiation_tensor)

        rho_legacy = see_legacy.get_solution_direct()
        rho = see.get_solution_direct()
        for level in term_registry.levels.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(level.L, level.S),
                Jʹ=TRIANGULAR(level.L, level.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                assert (
                    np.abs(rho(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ) - rho_legacy(level=level, K=K, Q=Q, J=J, Jʹ=Jʹ)).max()
                    < 1e-10
                )
