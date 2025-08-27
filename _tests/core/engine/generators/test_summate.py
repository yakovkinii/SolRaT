import unittest

from src.core.engine.functions.looping import FROMTO, PROJECTION, fromto
from src.core.engine.generators.summate import summate


class TestSummate(unittest.TestCase):
    def test_summate(self):
        p0 = 40
        R_summate = summate(
            lambda K, Q: K * abs(Q),
            K=f"range({p0+1})",
            Q="fromto(-K, K)",
        )

        R_summate2 = summate(
            lambda K, Q: K * abs(Q),
            K=FROMTO(0, p0),
            Q=PROJECTION("K"),
        )

        R = 0
        for k in range(p0 + 1):
            for q in fromto(-k, k):
                R += k * abs(q)

        assert R == R_summate
        assert R == R_summate2
