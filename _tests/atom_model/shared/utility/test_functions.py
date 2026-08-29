import unittest

import numpy as np

from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import (
    frequency_sm1_to_lambda_A,
    height_grid_refined_at_observer_surface,
    lambda_A_to_frequency_sm1,
    lambda_cm_to_frequency_sm1,
    reduced_frequency,
)


class TestWavelengthFrequency(unittest.TestCase):
    r"""
    Wavelength/frequency converters, checked by round-trip and against the definition.
    """

    def test_lambda_A_frequency_roundtrip(self):
        lambda_A = np.array([3000.0, 5000.0, 10000.0])
        assert np.allclose(frequency_sm1_to_lambda_A(lambda_A_to_frequency_sm1(lambda_A)), lambda_A)

    def test_lambda_cm_to_frequency(self):
        lambda_cm = 5000.0e-8
        assert np.isclose(lambda_cm_to_frequency_sm1(lambda_cm), c_cm_sm1 / lambda_cm)
        # Same physical wavelength expressed in cm and in Angstrom must give the same frequency.
        assert np.isclose(lambda_cm_to_frequency_sm1(1.0), lambda_A_to_frequency_sm1(1.0e8))


class TestReducedFrequency(unittest.TestCase):
    r"""
    Frequency in Doppler-width units.
    """

    def test_zero_at_line_center(self):
        nu0 = 5.0e14
        assert np.allclose(reduced_frequency(np.array([nu0]), nu0, 2.0e5), 0.0)

    def test_unit_at_one_doppler_width(self):
        nu0 = 5.0e14
        delta_v = 2.0e5
        delta_nu_D = nu0 * delta_v / c_cm_sm1
        assert np.isclose(reduced_frequency(np.array([nu0 + delta_nu_D]), nu0, delta_v)[0], 1.0)


class TestHeightGrid(unittest.TestCase):
    r"""
    Geometric height grid packed near the observer surface.
    """

    def test_grid_spans_and_is_sorted(self):
        thickness = 1000e5
        z = height_grid_refined_at_observer_surface(thickness, n_near_surface=10, n_interior=5)
        assert z.shape == (15,)
        assert np.all(np.diff(z) >= 0.0)  # ascending: z[0] deep, z[-1] observer surface
        assert np.isclose(z[0], 0.0)
        assert np.isclose(z[-1], thickness)


if __name__ == "__main__":
    unittest.main()
