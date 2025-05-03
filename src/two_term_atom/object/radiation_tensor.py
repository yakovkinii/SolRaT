from typing import Dict

from src.core.engine.functions.general import delta, half_int_to_str
from src.core.engine.functions.looping import projection
from src.two_term_atom.terms_levels_transitions.transition_registry import Transition, TransitionRegistry


class RadiationTensor:
    def __init__(self, transition_registry: TransitionRegistry):
        self.tensor: Dict[str, float] = {}
        self.transition_registry = transition_registry

    def fill_isotropic(self, value):
        for transition in self.transition_registry.transitions.values():
            for k in [0, 1, 2]:
                for q in projection(k):
                    key = self.get_key(transition=transition, k=k, q=q)
                    self.tensor[key] = value * delta(k, 0) * delta(q, 0)

    @staticmethod
    def get_key(transition: Transition, k: int, q: int):
        return f"{transition.transition_id}_k{half_int_to_str(k)}_q{half_int_to_str(q)}"

    def get(self, transition: Transition, k: int, q: int):
        return self.tensor[self.get_key(transition=transition, k=k, q=q)]
