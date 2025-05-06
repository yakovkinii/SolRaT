import unittest

from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry


class TestTermRegistry(unittest.TestCase):
    def test_term_registry(self):
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
            energy_cmm1=100,
        )
        term_registry.validate()
