import unittest

from src.engine.functions.general import delta, fact2
from src.engine.generators.multiply import multiply


class TestMultiply(unittest.TestCase):
    def test_multiply(self):
        def expensive_calculation(i, is_raw):
            assert is_raw or i == 0, "Did not short-circuit correctly"
            return fact2(i)

        for i in [0, 1]:
            R_multiply = multiply(
                lambda: delta(i, 0),
                lambda: expensive_calculation(i, is_raw=False),
            )

            R = delta(i, 0) * expensive_calculation(i, is_raw=True)

            assert R == R_multiply
