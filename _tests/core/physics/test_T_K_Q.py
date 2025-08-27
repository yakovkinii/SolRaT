import logging
import unittest

import numpy as np
from yatools import logging_config

from src.engine.functions.looping import FROMTO, PROJECTION
from src.engine.generators.nested_loops import nested_loops
from src.common.rotations import T_K_Q, T_K_Q_double_rotation, WignerD


class TestTKQ(unittest.TestCase):
    def test_T_K_Q(self):
        logging_config.init(logging.INFO)

        # A couple of table values
        assert T_K_Q(0, 0, 0, 1, 1, 1) == 1
        assert T_K_Q(0, 0, 1, 1, 1, 1) == 0
        assert T_K_Q(2, 2, 0, np.pi / 4, np.pi / 2, 1) == 2.651438096812267e-17 + 0.4330127018922193j
        assert T_K_Q(2, -2, 0, np.pi / 4, np.pi / 2, 1) == 2.651438096812267e-17 - 0.4330127018922193j

        # Test first rotation (Omega -> Solar Vertical) in double-rotation
        chi = np.pi / 5
        theta = np.pi / 7
        gamma = np.pi / 9

        D_inverse_omega = WignerD(alpha=-gamma, beta=-theta, gamma=-chi, K_max=2)
        D_magnetic = WignerD(alpha=0, beta=0, gamma=0, K_max=2)
        # T_K_Q_double_rotation
        for K, Q, stokes_component_index in nested_loops(
            K=FROMTO(0, 2),
            Q=PROJECTION("K"),
            stokes_component_index=FROMTO(0, 3),
        ):
            # from table, T_K_Q inverts the omega under the hood
            t_k_q_table = T_K_Q(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                chi=chi,
                theta=theta,
                gamma=gamma,
            )

            # from double rotation
            t_k_q_double_rotation = T_K_Q_double_rotation(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                D_inverse_omega=D_inverse_omega,
                D_magnetic=D_magnetic,
            )
            assert abs(t_k_q_table - t_k_q_double_rotation) < 1e-10

        # Test the second rotation (Solar Vertical -> B) in double-rotation
        chi_B = np.pi / 5
        theta_B = np.pi / 7

        D_inverse_omega = WignerD(alpha=0, beta=0, gamma=0, K_max=2)
        D_magnetic = WignerD(alpha=chi_B, beta=theta_B, gamma=0, K_max=2)
        # T_K_Q_double_rotation
        for K, Q, stokes_component_index in nested_loops(
            K=FROMTO(0, 2),
            Q=PROJECTION("K"),
            stokes_component_index=FROMTO(0, 3),
        ):
            # from table, T_K_Q inverts the omega under the hood. Need to provide inverted B angles here,
            # so that the final direction is correct: Solar Vertical -> B
            t_k_q_table = T_K_Q(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                chi=0,
                theta=-theta_B,
                gamma=-chi_B,
            )

            # from double rotation
            t_k_q_double_rotation = T_K_Q_double_rotation(
                K=K,
                Q=Q,
                stokes_component_index=stokes_component_index,
                D_inverse_omega=D_inverse_omega,
                D_magnetic=D_magnetic,
            )
            assert abs(t_k_q_table - t_k_q_double_rotation) < 1e-10
