import numpy as np
from numpy import pi, sqrt, real

from core.object.atmosphere_parameters import AtmosphereParameters
from core.utility.constant import h
from core.base.generator import summate, multiply, n_proj
from core.base.math import m1p
from core.steps.paschen_back import C
from core.utility.wigner_3j_6j_9j import wigner_3j, wigner_6j
from core.terms_levels_transitions.transition_registry import TransitionRegistry


class RadiativeTransferCoefficients:
    def __init__(self,atmosphere_parameters:AtmosphereParameters, transition_registry: TransitionRegistry, nu: np.ndarray):
        self.atmosphere_parameters: AtmosphereParameters = atmosphere_parameters
        self.transition_registry: TransitionRegistry = transition_registry
        self.nu = nu

    def calculate(self):
        ...

    def eta_a(self):
        """
        Reference:
        (7.47a)
        """
        for transition in self.transition_registry.transitions.values():
            level_upper = transition.level_upper
            level_lower = transition.level_lower
            Ll = level_lower.l
            Lu = level_upper.l
            S = level_lower.s
            B = self.atmosphere_parameters.magnetic_field_gauss
            N=1 # Todo
            T = 1 # Todo
            result = summate(
                lambda K, Q, Kl, Ql, jl, Jl, Jʹl, Jʹʹl, ju, Ju, Jʹu, Ml, Mʹl, Mu, q, qʹ: multiply(
                    lambda: h * self.nu / 4 / pi * N * n_proj(Ll) * transition.einstein_b_lu * sqrt(3 * n_proj(K, Kl)),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: m1p(1 + Jʹʹl - Ml + qʹ),
                    lambda: C(J=Jl, j=jl, level=level_lower, M=Ml, magnetic_field_gauss=B),
                    lambda: C(J=Jʹʹl, j=jl, level=level_lower, M=Ml, magnetic_field_gauss=B),
                    lambda: C(J=Ju, j=ju, level=level_upper, M=Mu, magnetic_field_gauss=B),
                    lambda: C(J=Jʹu, j=ju, level=level_upper, M=Mu, magnetic_field_gauss=B),
                    lambda: sqrt(n_proj(Jl, Jʹl, Ju, Jʹu)),
                    lambda: wigner_3j(Ju, Jl, 1, -Mu, Ml, -q),
                    lambda: wigner_3j(Jʹu, Jʹl, 1, -Mu, Mʹl, -qʹ),
                    lambda: wigner_3j(1, 1, K, 1, -qʹ, -Q),
                    lambda: wigner_3j(Jʹʹl, Jʹl, Kl, Ml, -Mʹl, -Ql),
                    lambda: wigner_6j(Lu, Ll, 1, Jl, Ju, S),
                    lambda: wigner_6j(Lu, Ll, 1, Jʹl, Jʹu, S),
                    lambda: real(T * ρ * Φ),
                ),
                K="range_inclusive(0, 2)",
                Q="projection(K)",
                K1="...",
                Ql="projection(Kl)",
                jl="...",
                Jl="...",
                Jʹl="...",
                Jʹʹl="...",
                ju="...",
                Ju="...",
                Jʹu="...",
                Ml="...",
                Mʹl="...",
                Mu="...",
                q="range_inclusive(-1, 1)",
                qʹ="range_inclusive(-1, 1)",
            )
