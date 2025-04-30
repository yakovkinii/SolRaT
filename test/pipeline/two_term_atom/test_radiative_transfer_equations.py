import logging
import unittest

import numpy as np
from yatools import logging_config

from core.object.atmosphere_parameters import AtmosphereParameters
from core.object.radiation_tensor import RadiationTensor
from core.radiative_transfer_equations import RadiativeTransferCoefficients
from core.statistical_equilibrium_equations import TwoTermAtom
from core.terms_levels_transitions.term_registry import TermRegistry
from core.terms_levels_transitions.transition_registry import TransitionRegistry
from core.utility.black_body import get_BP
from core.utility.einstein_coefficients import b_lu_from_b_ul_two_level_atom, b_ul_from_a_two_level_atom


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        logging_config.init(logging.INFO)

        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            l=0,
            s=0.5,
            j=0.5,
            energy_cmm1=200_000,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0.5,
            j=0.5,
            energy_cmm1=220_000,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0.5,
            j=1.5,
            energy_cmm1=221_000,
        )
        term_registry.validate()

        nu = np.arange(5e14, 7e14, 1e12)  # Hz
        a_ul = 0.7e8  # 1/s
        b_ul = b_ul_from_a_two_level_atom(a_ul=a_ul, nu=nu)
        b_lu = b_lu_from_b_ul_two_level_atom(b_ul=b_ul, j_u=1.5, j_l=0.5)

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=term_registry.get_level(beta="2p", l=1, s=0.5),
            level_lower=term_registry.get_level(beta="1s", l=0, s=0.5),
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
            n_frequencies=len(nu),
        )

        atom.add_all_equations()
        rho = atom.get_solution_direct()
        # for nu in np.arange(5.5e14, 7e14, 1e12):
        radiative_transfer_coefficients = RadiativeTransferCoefficients(
            atmosphere_parameters=atmosphere_parameters,
            transition_registry=transition_registry,
            nu=nu,
        )
        eta_s = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
        eta_s_analytic = radiative_transfer_coefficients.eta_s_analytic_resonance(rho=rho, stokes_component_index=0)

        # plot
        from matplotlib import pyplot as plt

        plt.plot(nu, eta_s, label=r"$\eta_s$")
        plt.plot(nu, eta_s_analytic, '--', label=r"$\eta_s$ (analytic test case)")
        # plt.plot(np.arange(5.5e14, 7e14, 2e12), eta_s_values)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"$\eta_s$")
        plt.title(r"$\eta_s$ vs Frequency")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
