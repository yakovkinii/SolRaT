import logging

from src.core.engine.functions.general import delta
from src.core.engine.functions.looping import FROMTO, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.core.engine.objects.container import Container
from src.two_term_atom.terms_levels_transitions.transition_registry import Transition, TransitionRegistry


class RadiationTensor(Container):
    def __init__(self, transition_registry: TransitionRegistry):
        """
        Radiation tensor J^K_Q (nu_ul).
        Here we assume that transitions are spread apart in frequency, so that we can assign a bijection of
        transition <-> nu_ul, and store J for each transition instead for clarity.
        I believe K is always <=2 by construction (see eg. 5.157) for electric-dipole transitions due to T tensor.
        """
        super().__init__()
        self.transition_registry = transition_registry

    def fill_isotropic(self, value):
        logging.warning("J should be a scalar, not vector. Or a vector of duplicated scalars, no nu-dependence")
        for transition in self.transition_registry.transitions.values():
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = value * delta(K, 0) * delta(Q, 0)

    def fill_NLTE_near_isotropic(self, value):
        logging.warning("J should be a scalar, not vector. Or a vector of duplicated scalars, no nu-dependence")
        self.fill_isotropic(value=value)
        for transition in self.transition_registry.transitions.values():
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] += value * 0.1 * (delta(Q, 1) - delta(Q, -1))

    def __call__(self, transition: Transition, K: int, Q: int):
        result = self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]
        return result

    def add(self, transition: Transition, K: int, Q: int, value):
        key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
        if key in self.data:
            self.data[key] += value
        else:
            self.data[key] = value
