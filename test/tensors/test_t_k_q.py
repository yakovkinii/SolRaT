import unittest

from core.tensors.t_k_q import t_k_q
import numpy as np


class TestMathUtils(unittest.TestCase):
    def test_t_k_q(self):
        assert t_k_q(0, 0, 0, 1, 1, 1) == 1
        assert t_k_q(0, 0, 1, 1, 1, 1) == 0
        assert (
            t_k_q(2, 2, 0, np.pi / 4, np.pi / 2, 1)
            == 2.651438096812267e-17 + 0.4330127018922193j
        )
        assert (
            t_k_q(2, -2, 0, np.pi / 4, np.pi / 2, 1)
            == 2.651438096812267e-17 - 0.4330127018922193j
        )

    def test_t_k_q_vector(self):
        inputs = np.array(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [2, 2, 0, np.pi / 4, np.pi / 2, 1],
                [2, -2, 0, np.pi / 4, np.pi / 2, 1],
            ]
        )
        output = t_k_q(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3],
            inputs[:, 4],
            inputs[:, 5],
        )
        benchmark = np.array(
            [
                1,
                0,
                2.651438096812267e-17 + 0.4330127018922193j,
                2.651438096812267e-17 - 0.4330127018922193j,
            ]
        )
        assert np.all(output == benchmark)


if __name__ == "__main__":
    unittest.main()
