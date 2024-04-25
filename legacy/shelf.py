def get_solution(self):  # Todo wrong normalization - trace should be weighted
    logging.error("RUNNING INCORRECT IMPLEMENTATION")
    u, s, vt = np.linalg.svd(self.matrix_builder.rho_matrix)
    index_min_singular_value = np.argmin(s)
    solution_vector = vt[index_min_singular_value, :]
    trace = sum([solution_vector[index] for index in self.matrix_builder.trace_indexes])
    if trace == 0:
        while index_min_singular_value > 1:
            index_min_singular_value = index_min_singular_value - 1
            logging.warning("Trace = 0. Decreasing index by 1")
            solution_vector = vt[index_min_singular_value, :]
            trace = sum(
                [solution_vector[index] for index in self.matrix_builder.trace_indexes]
            )
            if trace != 0:
                break
    solution_vector = solution_vector / trace
    return solution_vector


import logging
import unittest

import numpy as np
from numpy import sqrt, pi
from yatools import logging_config

from core.tensor.atmosphere_parameters import AtmosphereParameters
from core.tensor.radiation_tensor import RadiationTensor
from core.utility.black_body import get_BP
from core.utility.constant import c
from core.utility.einstein_coefficients import (
    b_ul_from_a_two_level_atom,
    b_lu_from_b_ul_two_level_atom,
)
from core.utility.math import m1p
from core.utility.wigner_3j_6j_9j import w6j
from pipeline.two_term_atom.statistical_equilibrium_equations import TwoTermAtom
from pipeline.two_term_atom.term_registry import TermRegistry
from pipeline.two_term_atom.transition_registry import TransitionRegistry


class TestStatisticalEquilibriumEquations(unittest.TestCase):
    def test_statistical_equilibrium_equations_resonance(self):
        # (10.126)
        logging_config.init(logging.INFO)

        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            l=0,
            s=0,
            j=0,
            energy=200_000,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0,
            j=1,
            energy=220_000,
        )
        term_registry.validate()

        nu = 20_000 * c  # Hz
        a_ul = 0.7e8  # 1/s
        b_ul = b_ul_from_a_two_level_atom(a_ul=a_ul, nu=nu)
        b_lu = b_lu_from_b_ul_two_level_atom(b_ul=b_ul, j_u=1, j_l=0)

        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=term_registry.get_level(beta="2p", l=1, s=0),
            level_lower=term_registry.get_level(beta="1s", l=0, s=0),
            einstein_a_ul=a_ul,
            einstein_b_ul=b_ul,
            einstein_b_lu=b_lu,
        )

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry)
        I0 = get_BP(nu=nu, T=500000)
        radiation_tensor.fill_isotropic(I0)
        atom = TwoTermAtom(
            term_registry=term_registry,
            transition_registry=transition_registry,
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor=radiation_tensor,
        )
        atom.options.append("disable_r_s")

        logging.warning(rt)

        atom.add_all_equations()

        rho_matrix = atom.matrix_builder.rho_matrix
        rho_matrix_abs = np.abs(rho_matrix)
        import pandas as pd

        mat = pd.DataFrame(rho_matrix_abs)
        mat.index = atom.matrix_builder.coherence_id_to_index.keys()
        mat.columns = atom.matrix_builder.coherence_id_to_index.keys()

        sol1 = atom.get_solution()
        sol2 = atom.get_solution_direct()
        logging.info(f"Solution vector : {abs(sol1)}")
        logging.info(f"Solution vector2: {abs(sol2)}")
        # Analytic:
        N_l = 1

        rt = radiation_tensor.get(
            transition=transition_registry.get_transition(
                level_upper=term_registry.get_level(beta="2p", l=1, s=0),
                level_lower=term_registry.get_level(beta="1s", l=0, s=0),
            ),
            k=0,
            q=0,
        )

        rho_u_0_0 = (
            N_l
            * sqrt(3 * 3 * 3)
            / 1
            * b_lu
            / a_ul
            * m1p(1 - 0 + 0 + 1 + 0 + 0)
            * w6j(1, 1, 0, 1, 1, 0)
            * w6j(1, 1, 0, 1, 1, 0)
            * rt
        )
        rho_l_0_0 = N_l
        trace = rho_l_0_0 + rho_u_0_0  # this is wrong, should be weighted
        rho_l_0_0 = rho_l_0_0 / trace
        rho_u_0_0 = rho_u_0_0 / trace
        a = 1
        logging.info([rho_l_0_0, rho_u_0_0])


if __name__ == "__main__":
    unittest.main()
