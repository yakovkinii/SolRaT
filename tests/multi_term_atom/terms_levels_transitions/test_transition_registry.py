import logging
import unittest

from yatools import logging_config

from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TestTransitionRegistry(unittest.TestCase):
    def test_transition_registry(self):
        logging_config.init(logging.INFO)

        level_registry = LevelRegistry()
        level_registry.register_level(
            beta="1s",
            L=0,
            S=0,
            J=0,
            energy_cmm1=100,
        )
        level_registry.register_level(
            beta="2p",
            L=1,
            S=0,
            J=1,
            energy_cmm1=200,
        )
        level_registry.validate()

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            term_upper=level_registry.get_term(beta="2p", L=1, S=0),
            term_lower=level_registry.get_term(beta="1s", L=0, S=0),
            einstein_a_ul=0.1,
            einstein_b_ul=0.1,
            einstein_b_lu=0.1,
        )
