import logging
import unittest

from numpy import sqrt
from yatools import logging_config

from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.physics.einstein_coefficients import b_lu_from_b_ul_two_term_atom, b_ul_from_a_ul_two_term_atom
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        # (10.126)
        logging_config.init(logging.INFO)

        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            L=0,
            S=0,
            J=0,
            energy_cmm1=200_000,
        )
        term_registry.register_term(
            beta="2p",
            L=1,
            S=0,
            J=1,
            energy_cmm1=220_000,
        )
        term_registry.validate()

        a_ul = 0.7e8  # 1/s
        b_ul = b_ul_from_a_ul_two_term_atom(a_ul_sm1=a_ul, nu_ul=6e14)
        b_lu = b_lu_from_b_ul_two_term_atom(b_ul=b_ul, Lu=1, Ll=0)

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=term_registry.get_level(beta="2p", L=1, S=0),
            level_lower=term_registry.get_level(beta="1s", L=0, S=0),
            einstein_a_ul=a_ul,
            einstein_b_ul=b_ul,
            einstein_b_lu=b_lu,
        )

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=500_00)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_w(h_arcsec=30)
        atom = TwoTermAtom(
            term_registry=term_registry,
            transition_registry=transition_registry,
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor=radiation_tensor,
            disable_r_s=True,
        )

        atom.add_all_equations()
        solution = atom.get_solution_direct()

        # Analytic:
        rt = radiation_tensor(
            transition=transition_registry.get_transition(
                level_upper=term_registry.get_level(beta="2p", L=1, S=0),
                level_lower=term_registry.get_level(beta="1s", L=0, S=0),
            ),
            K=0,
            Q=0,
        )

        rho_u_0_0 = b_lu / a_ul / sqrt(3) * rt
        trace = 1 + sqrt(3) * rho_u_0_0
        rho_l_0_0 = 1 / trace
        rho_u_0_0 = rho_u_0_0 / trace
        assert (
            abs(rho_l_0_0 - solution(level=term_registry.get_level(beta="1s", L=0, S=0), K=0, Q=0, J=0, Jʹ=0)) < 1e-15
        ).all()
        assert (
            abs(rho_u_0_0 - solution(level=term_registry.get_level(beta="2p", L=1, S=0), K=0, Q=0, J=1, Jʹ=1)) < 1e-15
        ).all()
