import unittest

import numpy as np

from src.core.physics.wigner_3j_6j_9j import _w3j, _w6j, _w9j


class TestMathUtils(unittest.TestCase):
    def test_w3j(self):
        assert _w3j(0, 0, 0, 0, 0, 0) == 1
        assert _w3j(2, 2, 4, 0, 0, 0) == np.sqrt(2 / 15)
        assert _w3j(2, 2, 2, 0, 0, 0) == 0
        assert _w3j(2, 2, 2, 2, -2, 0) == np.sqrt(1 / 6)
        assert _w3j(4, 4, 4, 0, 0, 0) == -np.sqrt(2 / 35)

    def test_w6j(self):
        assert _w6j(0, 0, 0, 0, 0, 0) == 1
        assert _w6j(2, 4, 2, 2, 4, 2) == 1 / 30
        assert abs(_w6j(2, 4, 2, 4, 2, 2) + np.sqrt(1 / 20)) < 1e-15
        assert _w6j(1, 2, 3, 2, 1, 2) == -1 / 6
        assert _w6j(8, 8, 4, 8, 8, 8) == -23 / 1386

    def test_w9j(self):
        assert _w9j(0, 0, 0, 0, 0, 0, 0, 0, 0) == 1
        assert abs(_w9j(8, 8, 10, 8, 8, 10, 8, 8, 8) - 1186 / 184041 * np.sqrt(1 / 7)) < 1e-15
        assert abs(_w9j(2, 4, 6, 2, 6, 4, 4, 6, 8) - 2 / 105 * np.sqrt(2 / 7)) < 1e-15
