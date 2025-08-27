import logging
import unittest

import numpy as np
from yatools import logging_config

from src.core.physics.wigner_3j_6j_9j import _w3j_doubled_argument, _w6j_doubled_argument, _w9j_doubled_argument


class TestMathUtils(unittest.TestCase):
    def test_w3j(self):
        logging_config.init(logging.INFO)

        assert _w3j_doubled_argument(0, 0, 0, 0, 0, 0) == 1
        assert _w3j_doubled_argument(2, 2, 4, 0, 0, 0) == np.sqrt(2 / 15)
        assert _w3j_doubled_argument(2, 2, 2, 0, 0, 0) == 0
        assert _w3j_doubled_argument(2, 2, 2, 2, -2, 0) == np.sqrt(1 / 6)
        assert _w3j_doubled_argument(4, 4, 4, 0, 0, 0) == -np.sqrt(2 / 35)

    def test_w6j(self):
        logging_config.init(logging.INFO)

        assert _w6j_doubled_argument(0, 0, 0, 0, 0, 0) == 1
        assert _w6j_doubled_argument(2, 4, 2, 2, 4, 2) == 1 / 30
        assert abs(_w6j_doubled_argument(2, 4, 2, 4, 2, 2) + np.sqrt(1 / 20)) < 1e-15
        assert _w6j_doubled_argument(1, 2, 3, 2, 1, 2) == -1 / 6
        assert _w6j_doubled_argument(8, 8, 4, 8, 8, 8) == -23 / 1386

    def test_w9j(self):
        logging_config.init(logging.INFO)

        assert _w9j_doubled_argument(0, 0, 0, 0, 0, 0, 0, 0, 0) == 1
        assert abs(_w9j_doubled_argument(8, 8, 10, 8, 8, 10, 8, 8, 8) - 1186 / 184041 * np.sqrt(1 / 7)) < 1e-15
        assert abs(_w9j_doubled_argument(2, 4, 6, 2, 6, 4, 4, 6, 8) - 2 / 105 * np.sqrt(2 / 7)) < 1e-15
