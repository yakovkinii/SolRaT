import logging
import unittest

import numpy as np
from yatools import logging_config

from src.core.physics.voigt_profile import voigt


class TestMathUtils(unittest.TestCase):
    def test_voigt(self):
        logging_config.init(logging.INFO)

        frequencies = np.linspace(-10, 10, 200)
        for a in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 5]:
            voigt_h = np.real(voigt(nu=frequencies, a=a))
            voigt_l = np.imag(voigt(nu=frequencies, a=a))
            assert np.min(voigt_h) > 0
            assert np.max(np.abs(voigt_h - voigt_h[::-1])) < 1e-12  # symmetrical wrt nu=0
            assert np.max(np.abs(voigt_l + voigt_l[::-1])) < 1e-12  # anti-symmetrical wrt nu=0
