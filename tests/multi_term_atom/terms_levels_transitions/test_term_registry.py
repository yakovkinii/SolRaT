import logging
import unittest

from yatools import logging_config

from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry


class TestLevelRegistry(unittest.TestCase):
    def test_level_registry(self):
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
            energy_cmm1=100,
        )
        level_registry.validate()
