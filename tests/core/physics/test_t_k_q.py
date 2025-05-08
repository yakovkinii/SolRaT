import logging
import unittest

import numpy as np
from yatools import logging_config

from src.core.physics.rotations import T_K_Q


class TestMathUtils(unittest.TestCase):
    def test_t_k_q(self):
        logging_config.init(logging.INFO)

        assert T_K_Q(0, 0, 0, 1, 1, 1) == 1
        assert T_K_Q(0, 0, 1, 1, 1, 1) == 0
        assert T_K_Q(2, 2, 0, np.pi / 4, np.pi / 2, 1) == 2.651438096812267e-17 + 0.4330127018922193j
        assert T_K_Q(2, -2, 0, np.pi / 4, np.pi / 2, 1) == 2.651438096812267e-17 - 0.4330127018922193j
