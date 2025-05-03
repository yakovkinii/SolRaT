import unittest

import numpy as np

from src.core.physics.rotation_tensor_t_k_q import t_k_q


class TestMathUtils(unittest.TestCase):
    def test_t_k_q(self):
        assert t_k_q(0, 0, 0, 1, 1, 1) == 1
        assert t_k_q(0, 0, 1, 1, 1, 1) == 0
        assert t_k_q(2, 2, 0, np.pi / 4, np.pi / 2, 1) == 2.651438096812267e-17 + 0.4330127018922193j
        assert t_k_q(2, -2, 0, np.pi / 4, np.pi / 2, 1) == 2.651438096812267e-17 - 0.4330127018922193j
