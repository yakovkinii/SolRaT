import logging
import unittest

import numpy as np
from numpy import sqrt

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestStatisticalEquilibriumEquationsResonance(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        # (10.126)
        setup_logging(logging.INFO)

        level_registry = LevelRegistry()
        level_registry.register_level(
            beta="1s",
            L=0,
            S=0,
            J=0,
            energy_cmm1=200_000,
        )
        level_registry.register_level(
            beta="2p",
            L=1,
            S=0,
            J=1,
            energy_cmm1=220_000,
        )
        level_registry.validate()

        a_ul = 0.7e8  # 1/s

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            term_upper=level_registry.get_term(beta="2p", L=1, S=0),
            term_lower=level_registry.get_term(beta="1s", L=0, S=0),
            einstein_a_ul_sm1=a_ul,
        )
        b_lu = transition_registry.get_transition(
            term_upper=level_registry.get_term(beta="2p", L=1, S=0),
            term_lower=level_registry.get_term(beta="1s", L=0, S=0),
        ).einstein_b_lu

        model = Models.multi_term_atom()
        model = model.configure(
            config=model.Config(
                level_registry=level_registry,
                transition_registry=transition_registry,
                atomic_mass_amu=4,
                reference_lambda_A_air=np.nan,
                disable_r_s=True,
            )
        )

        model_legacy = Models.multi_term_atom_legacy()
        model_legacy = model_legacy.configure(
            config=model_legacy.Config(
                level_registry=level_registry,
                transition_registry=transition_registry,
                atomic_mass_amu=4,
                reference_lambda_A_air=np.nan,
                disable_r_s=True,
            )
        )

        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=0,
            temperature_K=7000,
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(
            h_arcsec=30,
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see_legacy = model_legacy.StatisticalEquilibriumEquations.from_model_config(model_legacy.config)

        see_legacy.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        rho_legacy = see_legacy.get_solution()
        rho = see.get_solution()

        # Analytic:
        J00 = radiation_tensor.get(
            transition=transition_registry.get_transition(
                term_upper=level_registry.get_term(beta="2p", L=1, S=0),
                term_lower=level_registry.get_term(beta="1s", L=0, S=0),
            ),
            K=0,
            Q=0,
        )

        rho_u_0_0 = b_lu / a_ul / sqrt(3) * J00
        trace = 1 + sqrt(3) * rho_u_0_0
        rho_l_0_0 = 1 / trace
        rho_u_0_0 = rho_u_0_0 / trace
        assert (
            abs(
                rho_l_0_0
                - rho_legacy(term_id=level_registry.get_term(beta="1s", L=0, S=0).term_id, K=0, Q=0, J=0, Jʹ=0)
            )
            < 1e-15
        ).all()
        assert (
            abs(
                rho_u_0_0
                - rho_legacy(term_id=level_registry.get_term(beta="2p", L=1, S=0).term_id, K=0, Q=0, J=1, Jʹ=1)
            )
            < 1e-15
        ).all()
        assert (
            abs(rho_l_0_0 - rho(term_id=level_registry.get_term(beta="1s", L=0, S=0).term_id, K=0, Q=0, J=0, Jʹ=0))
            < 1e-15
        ).all()
        assert (
            abs(rho_u_0_0 - rho(term_id=level_registry.get_term(beta="2p", L=1, S=0).term_id, K=0, Q=0, J=1, Jʹ=1))
            < 1e-15
        ).all()
