import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging


def _build_model(collisions):
    r"""
    A J=0 -> J=1 resonance line as a multi-level atom, with optional collisions. Returns the model
    plus the upper level id and the transition id (for setting collisional rates).
    """
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="1s", J=0, energy_cmm1=0, g=1.0)
    level_registry.register_level(alpha="2p", J=1, energy_cmm1=20_000, g=1.2)

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="2p", J=1),
        level_lower=level_registry.get_level(alpha="1s", J=0),
        einstein_a_ul_sm1=1e7,
    )

    model = Models.multi_level_atom()
    model = model.configure(
        config=model.Config(
            level_registry=level_registry,
            transition_registry=transition_registry,
            atomic_mass_amu=4.0,
            reference_lambda_A_air=5000.0,
            collisions=collisions,
        )
    )
    upper_level_id = level_registry.get_level(alpha="2p", J=1).level_id
    transition_id = next(iter(transition_registry.transitions.keys()))
    return model, upper_level_id, transition_id


def _solve(model):
    r"""
    Solve the SEE once (Planck radiation at 8000 K, local temperature 5000 K) and return rho.
    """
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=5000.0
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_planck(temperature_K=8000.0)
    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=Angles()),
    )
    return see.get_solution()


class TestParametrizedCollisionsContainer(unittest.TestCase):
    def test_set_get_and_defaults(self):
        r"""
        Rates default to zero and round-trip through the setters.
        """
        setup_logging()
        collisions = ParametrizedCollisions()
        assert collisions.deexcitation_rate_sm1("missing") == 0.0
        assert collisions.depolarizing_rate_sm1("missing", 2) == 0.0
        collisions.set_deexcitation_rate("t0", 1.5e8)
        collisions.set_depolarizing_rate("lvl", 2, 3.0e7)
        assert collisions.deexcitation_rate_sm1("t0") == 1.5e8
        assert collisions.depolarizing_rate_sm1("lvl", 2) == 3.0e7

    def test_rejects_invalid(self):
        r"""
        Invalid rank / negative rates trip the assertions.
        """
        collisions = ParametrizedCollisions()
        with self.assertRaises(AssertionError):
            collisions.set_depolarizing_rate("lvl", 0, 1.0)  # K >= 1 required
        with self.assertRaises(AssertionError):
            collisions.set_deexcitation_rate("t0", -1.0)


class TestMultiLevelSEECollisionsSmoke(unittest.TestCase):
    def test_runs_and_finite(self):
        r"""
        The SEE with collisions (de-excitation + depolarizing) solves to a finite rho.
        """
        setup_logging()
        collisions = ParametrizedCollisions()
        model, upper_level_id, transition_id = _build_model(collisions)
        collisions.set_deexcitation_rate(transition_id, 5.0e7)
        collisions.set_depolarizing_rate(upper_level_id, 2, 2.0e7)

        rho = _solve(model)
        for value in rho.data.values():
            assert np.isfinite(np.real(value)) and np.isfinite(np.imag(value))
        assert np.real(rho(K=0, Q=0, level_id=upper_level_id)) >= 0

    def test_collisions_change_solution(self):
        r"""
        Enabling collisions (radiation temperature differs from the local temperature) changes the
        upper-level population relative to the collisionless case. One model is used; the mutable
        collisions object is set between the two solves, so both share the same level ids.
        """
        setup_logging()
        collisions = ParametrizedCollisions()
        model, upper_level_id, transition_id = _build_model(collisions)

        rho_free = _solve(model)  # all rates zero -> add_collisions contributes nothing
        collisions.set_deexcitation_rate(transition_id, 1.0e9)
        rho_coll = _solve(model)

        assert not np.isclose(
            np.real(rho_free(K=0, Q=0, level_id=upper_level_id)),
            np.real(rho_coll(K=0, Q=0, level_id=upper_level_id)),
        )


if __name__ == "__main__":
    unittest.main()
