import unittest

import numpy as np

# from core.utility.wigner_3j_6j_9j import w3j_doubled, w6j_doubled, w9j_doubled, _w9j


class TestMathUtils(unittest.TestCase):
    def test_w3j(self):
        assert w3j_doubled(0, 0, 0, 0, 0, 0) == 1
        assert w3j_doubled(2, 2, 4, 0, 0, 0) == np.sqrt(2 / 15)
        assert w3j_doubled(2, 2, 2, 0, 0, 0) == 0
        assert w3j_doubled(2, 2, 2, 2, -2, 0) == np.sqrt(1 / 6)
        assert w3j_doubled(4, 4, 4, 0, 0, 0) == -np.sqrt(2 / 35)

    def test_w3j_vector(self):
        inputs = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 2, 4, 0, 0, 0],
                [2, 2, 2, 0, 0, 0],
                [2, 2, 2, 2, -2, 0],
                [4, 4, 4, 0, 0, 0],
            ]
        )
        outputs = w3j_doubled(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3],
            inputs[:, 4],
            inputs[:, 5],
        )
        assert np.all(
            outputs
            == np.array([1, np.sqrt(2 / 15), 0, np.sqrt(1 / 6), -np.sqrt(2 / 35)])
        )

    def test_w6j(self):
        assert w6j_doubled(0, 0, 0, 0, 0, 0) == 1
        assert w6j_doubled(2, 4, 2, 2, 4, 2) == 1 / 30
        assert abs(w6j_doubled(2, 4, 2, 4, 2, 2) + np.sqrt(1 / 20)) < 1e-15
        assert w6j_doubled(1, 2, 3, 2, 1, 2) == -1 / 6
        assert w6j_doubled(8, 8, 4, 8, 8, 8) == -23 / 1386

    def test_w6j_vector(self):
        inputs = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 4, 2, 2, 4, 2],
                [2, 4, 2, 4, 2, 2],
                [1, 2, 3, 2, 1, 2],
                [8, 8, 4, 8, 8, 8],
            ]
        )
        outputs = w6j_doubled(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3],
            inputs[:, 4],
            inputs[:, 5],
        )
        assert (
            np.max(
                np.abs(
                    outputs
                    - np.array([1, 1 / 30, -1 / np.sqrt(20), -1 / 6, -23 / 1386])
                )
            )
            < 1e-15
        )

    def test_w9j(self):
        assert w9j_doubled(0, 0, 0, 0, 0, 0, 0, 0, 0) == 1
        assert (
            abs(_w9j(8, 8, 10, 8, 8, 10, 8, 8, 8) - 1186 / 184041 * np.sqrt(1 / 7))
            < 1e-15
        )
        assert abs(_w9j(2, 4, 6, 2, 6, 4, 4, 6, 8) - 2 / 105 * np.sqrt(2 / 7)) < 1e-15

    def test_w9j_vector(self):
        inputs = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 8, 10, 8, 8, 10, 8, 8, 8],
                [2, 4, 6, 2, 6, 4, 4, 6, 8],
            ]
        )
        outputs = w9j_doubled(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3],
            inputs[:, 4],
            inputs[:, 5],
            inputs[:, 6],
            inputs[:, 7],
            inputs[:, 8],
        )
        benchmark = np.array(
            [1, 1186 / 184041 * np.sqrt(1 / 7), 2 / 105 * np.sqrt(2 / 7)]
        )
        assert np.max(np.abs(outputs - benchmark)) < 1e-15


if __name__ == "__main__":
    unittest.main()
