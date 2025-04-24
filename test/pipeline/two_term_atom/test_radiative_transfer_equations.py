import logging
import unittest

from yatools import logging_config

from core.object.atmosphere_parameters import AtmosphereParameters
from core.object.radiation_tensor import RadiationTensor
from core.statistical_equilibrium_equations import TwoTermAtom
from core.utility.black_body import get_BP
from core.utility.constant import c
from core.utility.einstein_coefficients import (
    b_ul_from_a_two_level_atom,
    b_lu_from_b_ul_two_level_atom,
)
from core.radiative_transfer_equations import RadiativeTransferCoefficients
from core.terms_levels_transitions.term_registry import TermRegistry
from core.terms_levels_transitions.transition_registry import TransitionRegistry


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
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

        nu = 20_000 * c  # Hz
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
        )
        atom.options.append("disable_r_s")

        atom.add_all_equations()
        rho = atom.get_solution_direct()

        radiative_transfer_coefficients = RadiativeTransferCoefficients(
            atmosphere_parameters=atmosphere_parameters,
            transition_registry=transition_registry,
            nu=nu,
        )
        eta_a = radiative_transfer_coefficients.eta_a(rho=rho, stokes_component_index=2)
        logging.info(eta_a)


if __name__ == "__main__":
    unittest.main()
