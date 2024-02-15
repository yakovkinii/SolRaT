import unittest

from pipeline.two_term_atom.term_registry import TermRegistry


class TestTermRegistry(unittest.TestCase):
    def test_term_registry(self):
        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            l=0,
            s=0,
            j=0,
            energy=100,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0,
            j=1,
            energy=100,
        )
        term_registry.validate()


if __name__ == "__main__":
    unittest.main()
