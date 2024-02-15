import unittest

from pipeline.two_term_atom.term_registry import TermRegistry
from pipeline.two_term_atom.transition_registry import TransitionRegistry


class TestTransitionRegistry(unittest.TestCase):
    def test_transition_registry(self):
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
            energy=200,
        )
        term_registry.validate()

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=term_registry.get_level(beta="2p", l=1, s=0),
            level_lower=term_registry.get_level(beta="1s", l=0, s=0),
            einstein_a_ul=0.1,
            einstein_b_ul=0.1,
            einstein_b_lu=0.1,
        )


if __name__ == "__main__":
    unittest.main()
