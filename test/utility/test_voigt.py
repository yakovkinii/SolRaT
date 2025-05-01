import unittest

import numpy as np

from core.utility.voigt import voigt


class TestMathUtils(unittest.TestCase):
    def test_voigt(self):
        frequencies = np.linspace(-10, 10, 200)
        for a in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5]:
            a_values = np.ones_like(frequencies) * a
            voigt_h = np.real(voigt(nu=frequencies, a=a_values))
            voigt_l = np.imag(voigt(nu=frequencies, a=a_values))
            assert np.min(voigt_h) > 0
            assert np.max(np.abs(voigt_h - voigt_h[::-1])) < 1e-12  # symmetrical wrt nu=0
            assert np.max(np.abs(voigt_l + voigt_l[::-1])) < 1e-12  # anti-symmetrical wrt nu=0


if __name__ == "__main__":
    unittest.main()
