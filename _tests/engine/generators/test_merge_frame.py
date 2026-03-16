import unittest

import pandas as pd

from solrat.engine.functions.looping import FROMTO, INTERSECTION, TRIANGULAR, VALUE
from solrat.engine.generators.merge_frame import Frame, SumLimits
from solrat.engine.generators.merge_loopers import (
    ApplyConstraint,
    Constraint,
    DummyOrAlreadyMerged,
    Triangular,
)
from solrat.engine.generators.nested_loops import nested_loops


class TestMergeFrame(unittest.TestCase):
    def test_merge_frame(self):
        def f(J, Jʹ):
            # This is a non-physical function that simulates some complex operation using J and Jʹ
            return (J + 0.1) ** 5 + 0.37 * (Jʹ + 0.7) ** 3

        def g(L, Jʹ):
            # This is a non-physical function that simulates some complex operation using L and Jʹ
            return (L + 0.86) ** 5 + 0.37 * (Jʹ + 0.23) ** 3

        # We will calculate the sum of f(J, Jʹ)*g(L, Jʹ) over all:
        # 1) L=0..2,
        # 2) S=0.5,
        # 3) all triangular J limited to J=0.5,
        # 4) all triangular Jʹ.

        # ==================================================
        # 1. using Frame engine - most powerful and flexible
        # ==================================================

        base_frame = pd.DataFrame(
            {
                "L": [0, 1, 2],
                "S": [0.5, 0.5, 0.5],
                "J_constraint": [(0.5,), (0.5,), (0.5,)],  # tuples of allowed values
            }
        )

        class CustomSumLimits(SumLimits):
            L = DummyOrAlreadyMerged()
            S = DummyOrAlreadyMerged()
            J_constraint = Constraint()
            J = ApplyConstraint(Triangular(L, S), J_constraint)
            Jʹ = Triangular(L, S)

        limits = CustomSumLimits()
        frame = Frame.from_sum_limits(
            base_frame=base_frame,
            sum_limits=limits,
        )

        frame.register_multiplication(
            lambda J, Jʹ: f(J, Jʹ),  # can be positional for compactness
            g_factor=lambda L, Jʹ: g(L, Jʹ),  # can be keyword for easier debugging
        )

        result_frame0 = frame.copy().reduce()  # reduce in default order
        result_frame1 = frame.copy().reduce()  # to check that copy() works properly
        result_frame2 = frame.copy().reduce("L", "Jʹ", ...)  # first reduce L and Jʹ
        result_frame3 = frame.copy().reduce(..., "J", "Jʹ")  # or reduce J and Jʹ last
        result_frame4 = frame.copy().reduce(..., limits.J, limits.Jʹ)  # can use object fields
        intermediate_result = frame.reduce("J")  # reduce only J
        assert intermediate_result is None
        result_frame5 = frame.reduce()  # reduce the rest

        # ======================================================
        # 2. Using nested loops engine - clean but less flexible
        # ======================================================

        result_nested_loops = 0
        for L, S, J, Jʹ in nested_loops(
            L=FROMTO(0, 2),
            S=VALUE(0.5),
            J=INTERSECTION(TRIANGULAR("L", "S"), VALUE(0.5)),
            Jʹ=TRIANGULAR("L", "S"),
        ):
            result_nested_loops += f(J, Jʹ) * g(L, Jʹ)

        # ======================
        # 3. Using regular loops
        # ======================

        result_regular = 0
        for l2 in range(0, 2 * 2 + 1):  # l2 = 2L
            s2 = int(0.5 * 2)  # s2 = 2S
            for j2 in range(abs(l2 - s2), l2 + s2 + 1, 2):  # j2 = 2J
                if j2 != int(2 * 0.5):
                    continue
                for j2_prime in range(abs(l2 - s2), l2 + s2 + 1, 2):  # j2_prime = 2Jʹ
                    result_regular += f(j2 / 2, j2_prime / 2) * g(l2 / 2, j2_prime / 2)

        results = [
            result_frame0,
            result_frame1,
            result_frame2,
            result_frame3,
            result_frame4,
            result_frame5,
            result_nested_loops,
            result_regular,
        ]
        for result in results:
            assert abs(result - result_regular) < 1e-10
