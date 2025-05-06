import unittest

from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TestTransitionRegistry(unittest.TestCase):
    def test_transition_registry(self):
        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            L=0,
            S=0,
            J=0,
            energy_cmm1=100,
        )
        term_registry.register_term(
            beta="2p",
            L=1,
            S=0,
            J=1,
            energy_cmm1=200,
        )
        term_registry.validate()

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=term_registry.get_level(beta="2p", L=1, S=0),
            level_lower=term_registry.get_level(beta="1s", L=0, S=0),
            einstein_a_ul=0.1,
            einstein_b_ul=0.1,
            einstein_b_lu=0.1,
        )
