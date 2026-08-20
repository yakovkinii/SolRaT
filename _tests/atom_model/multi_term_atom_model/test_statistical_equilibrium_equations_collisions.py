import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging

TEMPERATURE_K = 6000.0


def build_two_J_upper_term(collisions):
    r"""
    A :math:`{}^2P_{1/2,3/2} \to {}^2S_{1/2}` atom (upper term carries two J levels) with an optional
    collisions object attached.
    """
    levels = LevelRegistry()
    levels.register_level(beta="lo", L=0, S=0.5, J=0.5, energy_cmm1=0.0)
    levels.register_level(beta="up", L=1, S=0.5, J=0.5, energy_cmm1=20000.0)
    levels.register_level(beta="up", L=1, S=0.5, J=1.5, energy_cmm1=20000.5)
    levels.validate()
    transitions = TransitionRegistry()
    transitions.register_transition(
        term_upper=levels.get_term(beta="up", L=1, S=0.5),
        term_lower=levels.get_term(beta="lo", L=0, S=0.5),
        einstein_a_ul_sm1=1.0e7,
    )
    config = MultiTermAtomConfig(
        level_registry=levels,
        transition_registry=transitions,
        atomic_mass_amu=40.0,
        reference_lambda_A_air=5000.0,
        collisions=collisions,
    )
    return Models.multi_term_atom().configure(config=config)


def solve(model, radiation_kind):
    r"""
    Solve the SEE once against an isotropic (``"iso"``) or anisotropic (``"aniso"``) field; B = 0.
    """
    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
    )
    radiation = model.RadiationTensor.from_model_config(model.config)
    if radiation_kind == "iso":
        radiation = radiation.fill_planck(temperature_K=TEMPERATURE_K)
    else:
        radiation = radiation.fill_NLTE_n_w_parametrized(h_arcsec=30)
    see.fill_all_equations(
        atmosphere_parameters=params,
        radiation_tensor_in_magnetic_frame=radiation.rotate_to_magnetic_frame(angles=Angles()),
    )
    return see.get_solution()


def fractional_alignment(rho, term_id):
    r"""
    ``rho^2_0/rho^0_0`` of the J = 3/2 upper level (K = 2 exists only where 2J >= 2).
    """
    rho00 = np.real(rho(0, 0, 1.5, 1.5, term_id))
    rho20 = np.real(rho(2, 0, 1.5, 1.5, term_id))
    return rho20 / rho00


class TestMultiTermMultiJCollisions(unittest.TestCase):
    r"""
    Collisions in the multi-term SEE on a genuinely multi-J term (upper :math:`{}^2P`, J = 1/2, 3/2),
    exercising the arbitrary-number-of-levels ``add_collisions`` path with two physical checks.
    """

    def setUp(self):
        setup_logging()

    def test_isotropic_field_has_no_alignment(self):
        collisions = ParametrizedCollisions()
        model = build_two_J_upper_term(collisions)
        transition = next(iter(model.config.transition_registry.transitions.values()))
        collisions.fill_deexcitation_from_epsilon(transition, 0.99, TEMPERATURE_K)  # C_ul >> A_ul
        alignment = fractional_alignment(solve(model, "iso"), transition.term_upper.term_id)
        self.assertLess(abs(alignment), 1e-8)  # isotropic illumination cannot align the level

    def test_strong_collisions_depolarize(self):
        collisions = ParametrizedCollisions()
        model = build_two_J_upper_term(collisions)
        transition = next(iter(model.config.transition_registry.transitions.values()))
        term_id = transition.term_upper.term_id

        # No rates set: add_collisions runs but every component rate is zero (pure scattering).
        scattering = fractional_alignment(solve(model, "aniso"), term_id)
        self.assertTrue(np.isfinite(scattering))
        self.assertGreater(abs(scattering), 0.0)

        # Strong inelastic collisions reduce the fractional alignment.
        collisions.fill_deexcitation_from_epsilon(transition, 0.99, TEMPERATURE_K)
        collisional = fractional_alignment(solve(model, "aniso"), term_id)
        self.assertTrue(np.isfinite(collisional))
        self.assertLess(abs(collisional), abs(scattering))


if __name__ == "__main__":
    unittest.main()
