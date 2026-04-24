import logging
import unittest

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestTransitionRegistry(unittest.TestCase):
    def test_transition_registry(self):
        setup_logging(logging.INFO)

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
            einstein_a_ul_sm1=0.1,
        )
