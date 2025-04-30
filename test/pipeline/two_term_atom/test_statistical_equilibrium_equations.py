import logging
import unittest

from numpy import sqrt
from yatools import logging_config

from core.object.atmosphere_parameters import AtmosphereParameters
from core.object.radiation_tensor import RadiationTensor
from core.statistical_equilibrium_equations import TwoTermAtom
from core.terms_levels_transitions.term_registry import TermRegistry
from core.terms_levels_transitions.transition_registry import TransitionRegistry
from core.utility.black_body import get_BP
from core.utility.constant import c
from core.utility.einstein_coefficients import b_lu_from_b_ul_two_level_atom, b_ul_from_a_two_level_atom


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        # (10.126)
        logging_config.init(logging.INFO)

        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            l=0,
            s=0,
            j=0,
            energy_cmm1=200_000,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0,
            j=1,
            energy_cmm1=220_000,
        )
        term_registry.validate()

        nu = 6.5e14  # Hz
        a_ul = 0.7e8  # 1/s
        b_ul = b_ul_from_a_two_level_atom(a_ul=a_ul, nu=nu)
        b_lu = b_lu_from_b_ul_two_level_atom(b_ul=b_ul, j_u=1, j_l=0)

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=term_registry.get_level(beta="2p", l=1, s=0),
            level_lower=term_registry.get_level(beta="1s", l=0, s=0),
            einstein_a_ul=a_ul,
            einstein_b_ul=b_ul,
            einstein_b_lu=b_lu,
        )

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry)
        I0 = get_BP(nu=nu, T=500000)
        radiation_tensor.fill_isotropic(I0)
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
        rt = radiation_tensor.get(
            transition=transition_registry.get_transition(
                level_upper=term_registry.get_level(beta="2p", l=1, s=0),
                level_lower=term_registry.get_level(beta="1s", l=0, s=0),
            ),
            k=0,
            q=0,
        )

        rho_u_0_0 = b_lu / a_ul / sqrt(3) * rt
        trace = 1 + sqrt(3) * rho_u_0_0
        rho_l_0_0 = 1 / trace
        rho_u_0_0 = rho_u_0_0 / trace
        assert (
            abs(rho_l_0_0 - solution(level=term_registry.get_level(beta="1s", l=0, s=0), K=0, Q=0, J=0, Jʹ=0)) < 1e-15
        )
        assert (
            abs(rho_u_0_0 - solution(level=term_registry.get_level(beta="2p", l=1, s=0), K=0, Q=0, J=1, Jʹ=1)) < 1e-15
        )


if __name__ == "__main__":
    unittest.main()
