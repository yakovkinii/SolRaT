import logging
import unittest

import numpy as np
from yatools import logging_config

from src.common.constants import h_erg_s, kB_erg_Km1, sqrt2
from src.multi_term_atom.atomic_data.mock import get_mock_atom_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.statistical_equilibrium_equations import (
    MultiTermAtomSEE,
    MultiTermAtomSEELTE,
)


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        # (10.126)
        logging_config.init(logging.INFO)

        level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu = get_mock_atom_data(
            fine_structure=False
        )

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=0,
            temperature_K=10000,
            atomic_mass_amu=atomic_mass_amu,
        )
        radiation_tensor = (
            RadiationTensor(transition_registry=transition_registry)
            .fill_planck(T_K=atmosphere_parameters.temperature_K)
            .rotate_to_magnetic_frame(
                chi_B=angles.chi_B,
                theta_B=angles.theta_B,
            )
        )

        # Set up the statistical equilibrium equations.
        # Do not pre-compute coefficients, load them from file instead for the purpose of this demo.
        see = MultiTermAtomSEE(
            level_registry=level_registry,
            transition_registry=transition_registry,
        )

        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
        )
        seelte = MultiTermAtomSEELTE(
            level_registry=level_registry,
        )

        rho = see.get_solution()
        rholte = seelte.get_solution(atmosphere_parameters=atmosphere_parameters)

        exp_hnu_kT = np.exp(
            -h_erg_s
            * next(iter(transition_registry.transitions.values())).get_mean_transition_frequency_sm1()
            / kB_erg_Km1
            / atmosphere_parameters.temperature_K
        )
        rho_analytical = {
            "1s_L=0.0_S=0.5_K=0.0_Q=0.0_J=0.5_Jʹ=0.5": 1 / sqrt2 / (1 + 3 * exp_hnu_kT),
            "2p_L=1.0_S=0.5_K=0.0_Q=0.0_J=0.5_Jʹ=0.5": exp_hnu_kT / sqrt2 / (1 + 3 * exp_hnu_kT),
            "2p_L=1.0_S=0.5_K=0.0_Q=0.0_J=1.5_Jʹ=1.5": exp_hnu_kT / (1 + 3 * exp_hnu_kT),
        }

        for coherence_id, coherence in rho.data.items():
            coherence_lte = rholte.data[coherence_id]
            if coherence_lte == 0:
                assert np.abs(coherence) < 1e-15
            else:
                assert np.abs((coherence_lte - coherence) / coherence_lte) < 1e-15
                if coherence_id in rho_analytical:
                    coherence_analytical = rho_analytical[coherence_id]
                    assert np.abs((coherence_lte - coherence_analytical) / coherence_analytical) < 1e-15
                    assert np.abs((coherence - coherence_analytical) / coherence_analytical) < 1e-15
