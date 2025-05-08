import logging
import unittest

import numpy as np
from yatools import logging_config

from src.core.engine.functions.looping import FROMTO, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.core.physics.rotations import T_K_Q, T_from_t, WignerD


class TestWignerD(unittest.TestCase):
    def test_wigner_D(self):
        """
        Compare T{K, Q} calculated from table 5.5 with T{K, Q} calculated from Wigner D function.
        This simultaneously tests t{K, P}, T{K, Q}, and Wigner D function.
        """
        logging_config.init(logging.INFO)

        chi = np.pi / 5
        theta = np.pi / 3
        gamma = np.pi / 7

        D = WignerD(alpha=chi, beta=theta, gamma=gamma, K_max=2)

        for K, Q, stokes_component_index in nested_loops(
            K=FROMTO(0, 2), Q=PROJECTION("K"), stokes_component_index=FROMTO(0, 3)
        ):
            T1 = T_from_t(K=K, Q=Q, stokes_component_index=stokes_component_index, D=D)
            T2 = T_K_Q(K=K, Q=Q, stokes_component_index=stokes_component_index, chi=chi, theta=theta, gamma=gamma)

            assert np.max(np.abs(T1 - T2)) < 1e-12
