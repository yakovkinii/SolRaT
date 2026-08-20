import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry as MultiLevelLevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.transition_registry import (
    TransitionRegistry as MultiLevelTransitionRegistry,
)
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry as MultiTermLevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import (
    TransitionRegistry as MultiTermTransitionRegistry,
)
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import frequencies_around_line_sm1
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.decorators import VERBOSE

UPPER_ENERGY_CMM1 = 20_000.0
EINSTEIN_A_UL_SM1 = 1.0e7
ATOMIC_MASS_AMU = 56.0
REFERENCE_LAMBDA_A_AIR = 5000.0
TEMPERATURE_K = 6000.0
EPSILON = 1e-2


def build_multi_term(collisions):
    r"""
    The :math:`^1S_0 \to {}^1P_1` line (:math:`S=0`, one J per term) as a multi-term atom.
    """
    levels = MultiTermLevelRegistry()
    levels.register_level(beta="lower", L=0, S=0, J=0, energy_cmm1=0.0)
    levels.register_level(beta="upper", L=1, S=0, J=1, energy_cmm1=UPPER_ENERGY_CMM1)
    levels.validate()
    transitions = MultiTermTransitionRegistry()
    transitions.register_transition(
        term_upper=levels.get_term(beta="upper", L=1, S=0),
        term_lower=levels.get_term(beta="lower", L=0, S=0),
        einstein_a_ul_sm1=EINSTEIN_A_UL_SM1,
    )
    config = MultiTermAtomConfig(
        level_registry=levels,
        transition_registry=transitions,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        reference_lambda_A_air=REFERENCE_LAMBDA_A_AIR,
        collisions=collisions,
    )
    return Models.multi_term_atom().configure(config=config)


def build_multi_level(collisions):
    r"""
    The same :math:`J=0 \to 1` resonance line as a multi-level atom (upper Lande factor :math:`g_u=1`).
    """
    levels = MultiLevelLevelRegistry()
    levels.register_level(alpha="lower", J=0, energy_cmm1=0.0, g=1.0)
    levels.register_level(alpha="upper", J=1, energy_cmm1=UPPER_ENERGY_CMM1, g=1.0)
    transitions = MultiLevelTransitionRegistry()
    transitions.register_transition(
        level_upper=levels.get_level(alpha="upper", J=1),
        level_lower=levels.get_level(alpha="lower", J=0),
        einstein_a_ul_sm1=EINSTEIN_A_UL_SM1,
    )
    config = MultiLevelAtomConfig(
        level_registry=levels,
        transition_registry=transitions,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        reference_lambda_A_air=REFERENCE_LAMBDA_A_AIR,
        collisions=collisions,
    )
    return Models.multi_level_atom().configure(config=config)


def log_depth_grid(z_max_cm, n_depth):
    r"""
    Height grid with the depth below the observer surface logarithmically spaced.
    """
    depth = np.logspace(np.log10(z_max_cm * 1e-6), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth)


def run(model, nu):
    r"""
    Fixed four-iteration self-consistent solve on a tiny grid (agreement holds at every iteration).
    """
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=log_depth_grid(1000e5, 6),
        temperature_K=TEMPERATURE_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=0.0,
        n_mu_quadrature=2,
        n_phi_quadrature=3,
        max_iterations=4,
        tolerance=1e-14,  # never reached: run a fixed 4 iterations
        ng_acceleration=False,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    return atmosphere, emergent


class TestMultiTermVsMultiLevelSelfConsistent(unittest.TestCase):
    r"""
    On an :math:`S=0` line the multi-term and multi-level atoms are formally identical, so the
    self-consistent NLTE solve -- with the photon-destruction probability set through the parametrized
    collisions on both -- must agree to numerical precision, anchoring the multi-term ``add_collisions``
    path to the (TB1999-validated) multi-level solver.
    """

    def test_agree_to_numerical_precision(self):
        setup_logging(VERBOSE)

        collisions_mt = ParametrizedCollisions()
        collisions_ml = ParametrizedCollisions()
        model_mt = build_multi_term(collisions_mt)
        model_ml = build_multi_level(collisions_ml)
        transition_mt = next(iter(model_mt.config.transition_registry.transitions.values()))
        transition_ml = next(iter(model_ml.config.transition_registry.transitions.values()))
        collisions_mt.fill_deexcitation_from_epsilon(transition_mt, EPSILON, TEMPERATURE_K)
        collisions_ml.set_deexcitation_rate_from_epsilon(transition_ml, EPSILON, TEMPERATURE_K)

        params = model_mt.AtmosphereParameters(
            model_config=model_mt.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
        )
        nu0 = transition_mt.get_mean_transition_frequency_sm1()
        nu = frequencies_around_line_sm1(nu0, params.delta_v_thermal_cm_sm1, half_width_doppler=2.0, step_doppler=1.0)

        atmosphere_mt, emergent_mt = run(model_mt, nu)
        atmosphere_ml, emergent_ml = run(model_ml, nu)

        upper_term_id = transition_mt.term_upper.term_id
        upper_level_id = transition_ml.level_upper.level_id
        alignment_mt = np.array(
            [
                np.real(rho(2, 0, 1.0, 1.0, upper_term_id)) / np.real(rho(0, 0, 1.0, 1.0, upper_term_id))
                for rho in atmosphere_mt.rho_grid
            ]
        )
        alignment_ml = np.array(
            [
                np.real(rho(K=2, Q=0, level_id=upper_level_id)) / np.real(rho(K=0, Q=0, level_id=upper_level_id))
                for rho in atmosphere_ml.rho_grid
            ]
        )

        self.assertTrue(np.all(np.isfinite(alignment_mt)))
        self.assertTrue(np.allclose(alignment_mt, alignment_ml, rtol=1e-6, atol=1e-10))
        for stokes in ("I", "Q", "U", "V"):
            self.assertTrue(
                np.allclose(getattr(emergent_mt, stokes), getattr(emergent_ml, stokes), rtol=1e-6, atol=1e-12)
            )


if __name__ == "__main__":
    unittest.main()
