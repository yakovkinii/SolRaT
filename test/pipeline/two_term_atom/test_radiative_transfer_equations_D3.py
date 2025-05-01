import logging
import unittest

import numpy as np
from yatools import logging_config

from core.base.converter import nu_hz_to_lambda_cm, lambda_cm_to_nu_hz
from core.object.atmosphere_parameters import AtmosphereParameters
from core.object.radiation_tensor import RadiationTensor
from core.radiative_transfer_equations import RadiativeTransferCoefficients
from core.statistical_equilibrium_equations import TwoTermAtom
from core.terms_levels_transitions.term_registry import TermRegistry
from core.terms_levels_transitions.transition_registry import TransitionRegistry
from core.utility.black_body import get_BP
from core.utility.constant import c
from core.utility.einstein_coefficients import b_lu_from_b_ul_two_level_atom, b_ul_from_a_two_level_atom


class TestRadiativeTransferEquationsD3(unittest.TestCase):
    def test_radiative_transfer_equations_d3(self):
        logging_config.init(logging.INFO)

        #  python -m pytest --capture=no --log-cli-level=INFO .\test_radiative_transfer_equations_D3.py
        term_registry = TermRegistry()
        term_registry.register_term(
            beta="2s3",
            l=0,
            s=1,
            j=1,
            energy_cmm1=159855.9726,
        )
        term_registry.register_term(
            beta="3s3",
            l=0,
            s=1,
            j=1,
            energy_cmm1=183236.7905,
        )
        term_registry.register_term(
            beta="2p3",
            l=1,
            s=1,
            j=0,
            energy_cmm1=169087.8291,
        )
        term_registry.register_term(
            beta="2p3",
            l=1,
            s=1,
            j=1,
            energy_cmm1=169086.8412,
        )
        term_registry.register_term(
            beta="2p3",
            l=1,
            s=1,
            j=2,
            energy_cmm1=169086.7647,
        )

        term_registry.register_term(
            beta="3p3",
            l=1,
            s=1,
            j=0,
            energy_cmm1=185564.8528,
        )
        term_registry.register_term(
            beta="3p3",
            l=1,
            s=1,
            j=1,
            energy_cmm1=185564.5817,
        )
        term_registry.register_term(
            beta="3p3",
            l=1,
            s=1,
            j=2,
            energy_cmm1=185564.5602,
        )
        term_registry.register_term(
            beta="3d3",
            l=2,
            s=1,
            j=1,
            energy_cmm1=186101.5908,
        )
        term_registry.register_term(
            beta="3d3",
            l=2,
            s=1,
            j=2,
            energy_cmm1=186101.5466,
        )
        term_registry.register_term(
            beta="3d3",
            l=2,
            s=1,
            j=3,
            energy_cmm1=186101.5440,
        )
        term_registry.validate()

        lambda_A = np.arange(5876, 5879, 0.1)
        lambda_cm = lambda_A * 1e-8  # nm
        nu = lambda_cm_to_nu_hz(lambda_cm)  # Hz
        transition_registry = TransitionRegistry()

        transition_registry.register_transition_from_a_ul(
            level_upper=term_registry.get_level(beta="2p3", l=1, s=1),
            level_lower=term_registry.get_level(beta="2s3", l=0, s=1),
            einstein_a_ul_sm1=3 * 1.022e7,
        )
        transition_registry.register_transition_from_a_ul(
            level_upper=term_registry.get_level(beta="3p3", l=1, s=1),
            level_lower=term_registry.get_level(beta="2s3", l=0, s=1),
            einstein_a_ul_sm1=3 * 9.478e6,
        )
        transition_registry.register_transition_from_a_ul(
            level_upper=term_registry.get_level(beta="3s3", l=0, s=1),
            level_lower=term_registry.get_level(beta="2p3", l=1, s=1),
            einstein_a_ul_sm1=3.080e6 + 9.259e6 + 1.540e7,
        )
        transition_registry.register_transition_from_a_ul(
            level_upper=term_registry.get_level(beta="3d3", l=2, s=1),
            level_lower=term_registry.get_level(beta="2p3", l=1, s=1),
            einstein_a_ul_sm1=3.920e7 + 5.290e7 + 2.940e7 + 7.060e7 + 1.760e7 + 1.960e6,
        )

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=20000, delta_v_thermal_cm_sm1=1_000_00)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry)
        I0 = get_BP(nu=nu, T=5000)
        radiation_tensor.fill_isotropic(I0)
        atom = TwoTermAtom(
            term_registry=term_registry,
            transition_registry=transition_registry,
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor=radiation_tensor,
            disable_r_s=True,
            disable_n=True,
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
        eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
        eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)
        eta_s_analytic = radiative_transfer_coefficients.eta_s_analytic_resonance(rho=rho, stokes_component_index=0)

        # plot
        from matplotlib import pyplot as plt
        #
        plt.plot(lambda_A-5877.23, eta_sI, label=r"$\eta_s I$")
        plt.plot(lambda_A-5877.23, eta_sV, "-.", label=r"$\eta_s V$")
        plt.plot(lambda_A-5877.23, eta_s_analytic, "--", label=r"$\eta_s$ (analytic test case, no magnetic field)")
        # plt.plot(np.arange(5.5e14, 7e14, 2e12), eta_s_values)
        plt.xlabel(r"$\Delta\lambda$ ($\AA$)")
        plt.ylabel(r"$\eta_s$")
        plt.title(r"He I D3: $\eta_s$ vs $\Delta\lambda$")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
