import logging
import unittest

from yatools import logging_config

from src.two_term_atom.atomic_data.mock import get_mock_atom_data
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        """
        Test that SEE are runnable with stimulated emission enabled
        """
        logging_config.init(logging.INFO)

        term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=500_00)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_w(h_arcsec=30)
        atom = TwoTermAtom(
            term_registry=term_registry,
            transition_registry=transition_registry,
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor=radiation_tensor,
            disable_r_s=False,
        )

        atom.add_all_equations()
        _ = atom.get_solution_direct()
