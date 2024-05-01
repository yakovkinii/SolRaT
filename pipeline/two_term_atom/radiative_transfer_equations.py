import numpy as np
from numpy import pi, sqrt

from core.utility.constant import h
from core.utility.math import m1p
from core.utility.python import projection, range_inclusive
from pipeline.two_term_atom.transition_registry import TransitionRegistry


class RadiativeTransferCoefficients:
    def __init__(self):
        self.transition_registry: TransitionRegistry = ...
        self.nu = np.array([...])

    def calculate(self):
        ...

    def eta_a(self):
        m0 = h * self.nu / 4 / pi
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            l_l = level_lower.l
            l_u = level_upper.l
            s = level_lower.s

            m1 = (2 * l_l + 1) * transition.einstein_b_lu
            for k in range_inclusive(0, 2):
                for q in projection(k):
                    for k_l in ...:
                        for q_l in projection(k_l):
                            m2 = sqrt(3 * (2 * k + 1) * (2 * k_l + 1))
                            for j_small_l in ...:
                                for j_l in ...:
                                    for j_prime_l in ...:
                                        for j_prime_prime_l in ...:
                                            for j_small_u in ...:
                                                for j_u in ...:
                                                    for j_prime_u in ...:
                                                        for m_l in ...:
                                                            for m_prime_l in ...:
                                                                for m_u in ...:
                                                                    for q_small in [
                                                                        -1,
                                                                        0,
                                                                        1,
                                                                    ]:
                                                                        for q_small_prime in (
                                                                            [-1, 0, 1]
                                                                        ):
                                                                            m3 = m1p(
                                                                                1
                                                                                + j_prime_prime_l
                                                                                - m_l
                                                                                + q_small_prime
                                                                            )
                                                                            m4 = ...
