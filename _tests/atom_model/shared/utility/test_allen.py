import unittest

import numpy as np

from solrat.atom_model.shared.utility.allen import geometric_factors, i0_allen


class TestAllenContinuum(unittest.TestCase):
    r"""
    Allen continuum intensity and the height-dilution geometry factors.
    """

    def test_i0_vanishes_at_the_limb_tangent(self):
        assert i0_allen(lambda_A=5000.0, mu=0.0) == 0.0

    def test_i0_positive_at_disk_center(self):
        value = i0_allen(lambda_A=5000.0, mu=1.0)
        assert np.isfinite(value) and value > 0.0

    def test_geometric_factors_at_surface(self):
        a0, a1, a2, b0, b1, b2 = geometric_factors(0.0)
        assert (a0, a1, a2) == (1.0, -0.5, -2.0 / 3.0)
        assert (b0, b1, b2) == (1.0 / 3.0, -1.0 / 12.0, -2.0 / 15.0)

    def test_geometric_factors_above_surface(self):
        factors = geometric_factors(960.0)  # ~1 solar radius above the surface
        assert len(factors) == 6
        assert all(np.isfinite(f) for f in factors)
        assert 0.0 < factors[0] < 1.0  # a0 = 1 - cos(gamma) is diluted below the surface value


if __name__ == "__main__":
    unittest.main()
