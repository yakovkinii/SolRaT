import unittest

import numpy as np
from numpy import sqrt

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry as MTLevelRegistry
from solrat.atom_model.multi_term_atom_model.object.transition_registry import (
    TransitionRegistry as MTTransitionRegistry,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestMultiLevelStatisticalEquilibriumEquationsResonance(unittest.TestCase):
    def test_two_level_resonance_analytical(self):
        """
        Two-level multi-level atom (J_l=0, J_u=1) under an isotropic Planck radiation field.
        Compare against the analytical resonance solution (LL04 10.126).
        """
        setup_logging()

        level_registry = LevelRegistry()
        level_registry.register_level(alpha="1s", J=0, energy_cmm1=200_000, g=1.0)
        level_registry.register_level(alpha="2p", J=1, energy_cmm1=220_000, g=1.0)

        a_ul = 0.7e8  # 1/s
        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=level_registry.get_level(alpha="2p", J=1),
            level_lower=level_registry.get_level(alpha="1s", J=0),
            einstein_a_ul_sm1=a_ul,
        )
        b_lu = transition_registry.get_transition(
            level_upper=level_registry.get_level(alpha="2p", J=1),
            level_lower=level_registry.get_level(alpha="1s", J=0),
        ).einstein_b_lu

        model = Models.multi_level_atom().configure(
            config=Models.multi_level_atom().Config(
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
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(
            h_arcsec=30,
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        rho = see.get_solution()

        # Analytical (LL04 10.126):  rho^0_0(upper) / rho^0_0(lower) = B_lu / A_ul / sqrt(2J_u + 1) * J^0_0
        # With trace-normalization sum sqrt(2J+1) rho^0_0(level) = 1.
        J00 = radiation_tensor.get(
            transition=transition_registry.get_transition(
                level_upper=level_registry.get_level(alpha="2p", J=1),
                level_lower=level_registry.get_level(alpha="1s", J=0),
            ),
            K=0,
            Q=0,
        )

        rho_u_00 = b_lu / a_ul / sqrt(3) * J00
        trace = 1 + sqrt(3) * rho_u_00
        rho_l_00 = 1 / trace
        rho_u_00 = rho_u_00 / trace

        l_lower = level_registry.get_level(alpha="1s", J=0)
        l_upper = level_registry.get_level(alpha="2p", J=1)

        assert abs(rho_l_00 - rho(level_id=l_lower.level_id, K=0, Q=0)) < 1e-15
        assert abs(rho_u_00 - rho(level_id=l_upper.level_id, K=0, Q=0)) < 1e-15

    def test_matches_multi_term_at_S_zero(self):
        """
        For an :math:`S=0` LS multiplet each multi-term term has only one :math:`J = L`,
        so the multi-level and multi-term descriptions coincide.  Solve SEE in both
        representations for the same atom/atmosphere and check the populations agree.
        """
        setup_logging()

        # Multi-term version (S = 0, J = L per term)
        a_ul = 1e7

        mt_level_registry = MTLevelRegistry()
        mt_level_registry.register_level(beta="1s", L=0, S=0, J=0, energy_cmm1=200_000)
        mt_level_registry.register_level(beta="2p", L=1, S=0, J=1, energy_cmm1=220_000)
        mt_level_registry.validate()

        mt_transition_registry = MTTransitionRegistry()
        mt_transition_registry.register_transition(
            term_upper=mt_level_registry.get_term(beta="2p", L=1, S=0),
            term_lower=mt_level_registry.get_term(beta="1s", L=0, S=0),
            einstein_a_ul_sm1=a_ul,
        )

        mt_model = Models.multi_term_atom().configure(
            config=Models.multi_term_atom().Config(
                level_registry=mt_level_registry,
                transition_registry=mt_transition_registry,
                atomic_mass_amu=4.0,
                reference_lambda_A_air=np.nan,
                disable_r_s=False,
            )
        )

        # Multi-level version (Lande factor for L=1, S=0, J=1 is 1)
        ml_level_registry = LevelRegistry()
        ml_level_registry.register_level(alpha="1s", J=0, energy_cmm1=200_000, g=1.0)
        ml_level_registry.register_level(alpha="2p", J=1, energy_cmm1=220_000, g=1.0)

        ml_transition_registry = TransitionRegistry()
        ml_transition_registry.register_transition(
            level_upper=ml_level_registry.get_level(alpha="2p", J=1),
            level_lower=ml_level_registry.get_level(alpha="1s", J=0),
            einstein_a_ul_sm1=a_ul,
        )

        ml_model = Models.multi_level_atom().configure(
            config=Models.multi_level_atom().Config(
                level_registry=ml_level_registry,
                transition_registry=ml_transition_registry,
                atomic_mass_amu=4.0,
                reference_lambda_A_air=np.nan,
                disable_r_s=False,
            )
        )

        # Same atmosphere
        mt_atm = mt_model.AtmosphereParameters(
            model_config=mt_model.config,
            magnetic_field_gauss=0,
            temperature_K=7000,
        )
        ml_atm = ml_model.AtmosphereParameters(
            model_config=ml_model.config,
            magnetic_field_gauss=0,
            temperature_K=7000,
        )

        mt_rad = mt_model.RadiationTensor.from_model_config(mt_model.config).fill_NLTE_n_w_allen(h_arcsec=30)
        ml_rad = ml_model.RadiationTensor.from_model_config(ml_model.config).fill_NLTE_n_w_allen(h_arcsec=30)

        mt_see = mt_model.StatisticalEquilibriumEquations.from_model_config(mt_model.config)
        mt_see.fill_all_equations(atmosphere_parameters=mt_atm, radiation_tensor_in_magnetic_frame=mt_rad)
        mt_rho = mt_see.get_solution()

        ml_see = ml_model.StatisticalEquilibriumEquations.from_model_config(ml_model.config)
        ml_see.fill_all_equations(atmosphere_parameters=ml_atm, radiation_tensor_in_magnetic_frame=ml_rad)
        ml_rho = ml_see.get_solution()

        # Compare populations.  Both models share the same trace normalization.
        mt_lower_term = mt_level_registry.get_term(beta="1s", L=0, S=0)
        mt_upper_term = mt_level_registry.get_term(beta="2p", L=1, S=0)
        ml_lower = ml_level_registry.get_level(alpha="1s", J=0)
        ml_upper = ml_level_registry.get_level(alpha="2p", J=1)

        assert (
            abs(
                mt_rho(term_id=mt_lower_term.term_id, K=0, Q=0, J=0, Jʹ=0)
                - ml_rho(level_id=ml_lower.level_id, K=0, Q=0)
            )
            < 1e-12
        )
        assert (
            abs(
                mt_rho(term_id=mt_upper_term.term_id, K=0, Q=0, J=1, Jʹ=1)
                - ml_rho(level_id=ml_upper.level_id, K=0, Q=0)
            )
            < 1e-12
        )


if __name__ == "__main__":
    unittest.main()
