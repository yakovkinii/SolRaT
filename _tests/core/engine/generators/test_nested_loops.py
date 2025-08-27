import unittest

from src.core.engine.functions.looping import FROMTO, fromto
from src.core.engine.generators.nested_loops import nested_loops


class TestNestedLoops(unittest.TestCase):
    def test_nested_loops(self):
        p0 = 40
        R_nested = 0
        R = 0

        for k, q in nested_loops(k=FROMTO(0, p0 - 1), q=FROMTO("-k", "k")):
            R_nested += k * abs(q)

        for k in range(p0):
            for q in fromto(-k, k):
                R += k * abs(q)

        assert R == R_nested
