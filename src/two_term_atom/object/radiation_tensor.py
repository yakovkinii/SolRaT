from src.core.engine.functions.general import delta
from src.core.engine.functions.looping import FROMTO, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.core.engine.objects.container import Container
from src.two_term_atom.terms_levels_transitions.transition_registry import Transition, TransitionRegistry


class RadiationTensor(Container):
    def __init__(self, transition_registry: TransitionRegistry):
        super().__init__()
        self.transition_registry = transition_registry

    def fill_isotropic(self, value):
        for transition in self.transition_registry.transitions.values():
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = value * delta(K, 0) * delta(Q, 0)

    def get(self, transition: Transition, K: int, Q: int):
        return self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]
