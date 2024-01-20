import unittest

import numpy as np

from core.utility.math import m1p


class TestMathUtils(unittest.TestCase):
    def test_m1p(self):
        assert m1p(0) == 1
        assert m1p(1) == -1
        assert m1p(-1) == -1
        assert m1p(100) == 1
        assert m1p(101) == -1
        assert np.all(m1p(np.array([-2, -1, 0, 1, 2])) == np.array([1, -1, 1, -1, 1]))


if __name__ == "__main__":
    unittest.main()
