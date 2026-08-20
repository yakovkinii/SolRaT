import unittest

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry


def multi_term_transition_two_lower_levels():
    r"""
    A :math:`{}^2D \to {}^2P` transition (upper J = 3/2, 5/2; lower J = 1/2, 3/2), so the lower term has
    two fine-structure levels and ``fill_deexcitation_from_epsilon`` splits the rate over them.
    """
    levels = LevelRegistry()
    levels.register_level(beta="lo", L=1, S=0.5, J=0.5, energy_cmm1=0.0)
    levels.register_level(beta="lo", L=1, S=0.5, J=1.5, energy_cmm1=100.0)
    levels.register_level(beta="up", L=2, S=0.5, J=1.5, energy_cmm1=20000.0)
    levels.register_level(beta="up", L=2, S=0.5, J=2.5, energy_cmm1=20000.5)
    levels.validate()
    transitions = TransitionRegistry()
    transitions.register_transition(
        term_upper=levels.get_term(beta="up", L=2, S=0.5),
        term_lower=levels.get_term(beta="lo", L=1, S=0.5),
        einstein_a_ul_sm1=1.0e7,
    )
    return next(iter(transitions.transitions.values()))


class TestComponentKey(unittest.TestCase):
    def test_component_key_format(self):
        self.assertEqual(ParametrizedCollisions.component_key("t", 1.0, 0.0), "t|Ju=1.0|Jl=0.0")
        self.assertEqual(ParametrizedCollisions.component_key("t", 1.5, 0.5), "t|Ju=1.5|Jl=0.5")


class TestMultiLevelSetterUnchanged(unittest.TestCase):
    def test_ml_keys_by_transition_id(self):
        model = PreconfiguredModels.multi_level_atom_mock()
        transition = next(iter(model.config.transition_registry.transitions.values()))
        collisions = ParametrizedCollisions()
        collisions.set_deexcitation_rate_from_epsilon(transition, 1e-2, 6000.0)
        self.assertGreater(collisions.deexcitation_rate_sm1(transition.transition_id), 0.0)


class TestMultiTermPerComponent(unittest.TestCase):
    def setUp(self):
        self.transition = multi_term_transition_two_lower_levels()

    def test_multi_term_requires_J(self):
        collisions = ParametrizedCollisions()
        with self.assertRaises(ValueError):
            collisions.set_deexcitation_rate_from_epsilon(self.transition, 1e-2, 6000.0)

    def test_set_single_component(self):
        collisions = ParametrizedCollisions()
        collisions.set_deexcitation_rate_from_epsilon(self.transition, 1e-2, 6000.0, J_upper=1.5, J_lower=0.5)
        set_key = ParametrizedCollisions.component_key(self.transition.transition_id, 1.5, 0.5)
        other_key = ParametrizedCollisions.component_key(self.transition.transition_id, 2.5, 1.5)
        self.assertGreater(collisions.deexcitation_rate_sm1(set_key), 0.0)
        self.assertEqual(collisions.deexcitation_rate_sm1(other_key), 0.0)

    def test_fill_distributes_over_lower_levels(self):
        # Reference: the full multiplet C_ul, from setting one component directly (no split).
        reference = ParametrizedCollisions()
        reference.set_deexcitation_rate_from_epsilon(self.transition, 1e-2, 6000.0, J_upper=1.5, J_lower=0.5)
        c_ul_total = reference.deexcitation_rate_sm1(
            ParametrizedCollisions.component_key(self.transition.transition_id, 1.5, 0.5)
        )
        n_lower = len(self.transition.term_lower.levels)
        self.assertEqual(n_lower, 2)

        filled = ParametrizedCollisions()
        filled.fill_deexcitation_from_epsilon(self.transition, 1e-2, 6000.0)
        for level_upper in self.transition.term_upper.levels:
            total = 0.0
            for level_lower in self.transition.term_lower.levels:
                rate = filled.deexcitation_rate_sm1(
                    ParametrizedCollisions.component_key(self.transition.transition_id, level_upper.J, level_lower.J)
                )
                self.assertAlmostEqual(rate / (c_ul_total / n_lower), 1.0, places=6)  # each component = C_ul/n_l
                total += rate
            self.assertAlmostEqual(total / c_ul_total, 1.0, places=6)  # total per upper level = C_ul

    def test_fill_rejects_multi_level(self):
        model = PreconfiguredModels.multi_level_atom_mock()
        transition = next(iter(model.config.transition_registry.transitions.values()))
        collisions = ParametrizedCollisions()
        with self.assertRaises(AssertionError):
            collisions.fill_deexcitation_from_epsilon(transition, 1e-2, 6000.0)


class TestDepolarizingRates(unittest.TestCase):
    def test_set_get_and_guards(self):
        collisions = ParametrizedCollisions()
        collisions.set_depolarizing_rate("lvl", 2, 3.0e6)
        self.assertEqual(collisions.depolarizing_rate_sm1("lvl", 2), 3.0e6)
        self.assertEqual(collisions.depolarizing_rate_sm1("lvl", 1), 0.0)  # unset
        self.assertEqual(collisions.depolarizing_rate_sm1("missing", 2), 0.0)  # unknown level
        with self.assertRaises(AssertionError):
            collisions.set_depolarizing_rate("lvl", 0, 1.0)  # K = 0 not allowed
        with self.assertRaises(AssertionError):
            collisions.set_depolarizing_rate("lvl", 2, -1.0)  # negative


if __name__ == "__main__":
    unittest.main()
