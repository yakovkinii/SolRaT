import unittest

from src.core.engine.functions.general import delta, fact
from src.core.engine.generators.multiply import multiply


class TestMultiply(unittest.TestCase):
    def test_summate(self):
        def expensive_calculation(i, is_raw):
            if not is_raw and i != 0:
                raise ValueError("Did not short-circuit correctly")
            return fact(i)

        for i in [0, 1]:
            R_multiply = multiply(
                lambda: delta(i, 0),
                lambda: expensive_calculation(i, is_raw=False),
            )

            R = delta(i, 0) * expensive_calculation(i, is_raw=True)

            assert R == R_multiply
