import unittest

from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry


class TestTermRegistry(unittest.TestCase):
    def test_term_registry(self):
        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            l=0,
            s=0,
            j=0,
            energy_cmm1=100,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0,
            j=1,
            energy_cmm1=100,
        )
        term_registry.validate()

