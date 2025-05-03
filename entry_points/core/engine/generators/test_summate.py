import unittest

from src.core.engine.functions.looping import FROMTO, fromto
from src.core.engine.generators.summate import summate


class TestSummate(unittest.TestCase):
    def test_summate(self):
        p0 = 40
        R = 0
        R_summate = summate(
            lambda K, Q: K * abs(Q),
            K=f"range({p0})",
            Q="fromto(-K, K)",
        )

        R_summate2 = summate(
            lambda K, Q: K * abs(Q),
            K=FROMTO(0, p0 - 1),
            Q=FROMTO("-K", "K"),
        )

        for k in range(p0):
            for q in fromto(-k, k):
                R += k * abs(q)

        assert R == R_summate
        assert R == R_summate2
