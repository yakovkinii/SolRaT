import logging
import unittest

import numpy as np
from yatools import logging_config

from src.engine.functions.looping import PROJECTION, TRIANGULAR
from src.engine.generators.nested_loops import nested_loops
from src.multi_term_atom.atomic_data.mock import get_mock_atom_data
from src.multi_term_atom.legacy.statistical_equilibrium_equations_legacy import (
    MultiTermAtomSEELegacy,
)
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        """
        Test that SEE are runnable with stimulated emission enabled
        """
        logging_config.init(logging.INFO)

        (
            level_registry,
            transition_registry,
            reference_lambda_A,
            reference_nu_sm1,
            atomic_mass_amu,
        ) = get_mock_atom_data()

        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=0, temperature_K=7000, atomic_mass_amu=atomic_mass_amu
        )
        radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(
            h_arcsec=30
        )
        see_legacy = MultiTermAtomSEELegacy(
            level_registry=level_registry,
            transition_registry=transition_registry,
            disable_r_s=False,
        )

        see = MultiTermAtomSEE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            disable_r_s=False,
        )

        see_legacy.add_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )

        rho_legacy = see_legacy.get_solution_direct()
        rho = see.get_solution()
        for term in level_registry.terms.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                assert (
                    np.abs(
                        rho(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                        - rho_legacy(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                    ).max()
                    < 1e-10
                )
