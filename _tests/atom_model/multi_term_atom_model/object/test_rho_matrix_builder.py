import unittest

import numpy as np

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.rho_matrix_builder import Rho


def build_terms():
    r"""
    A single ``^1S_0 -> ^1P_1`` term pair, returned as the list of :class:`Term` objects.
    """
    registry = LevelRegistry()
    registry.register_level(beta="lower", L=0, S=0, J=0, energy_cmm1=0.0)
    registry.register_level(beta="upper", L=1, S=0, J=1, energy_cmm1=20000.0)
    registry.validate()
    return list(registry.terms.values())


class TestRhoGetVector(unittest.TestCase):
    r"""
    The vectorized ``get_vector`` lookup must agree with the scalar ``__call__`` path for stored
    coherences and return NaN for absent ones (left-merge miss).
    """

    def test_vector_matches_scalar_and_missing_is_nan(self):
        terms = build_terms()
        upper = next(term for term in terms if term.L == 1)
        term_id = upper.term_id

        rho = Rho(terms=terms)
        rho.set_from_term_id(term_id=term_id, K=0, Q=0, J=1, Jʹ=1, value=1.5 + 0.5j)
        rho.set_from_term_id(term_id=term_id, K=2, Q=0, J=1, Jʹ=1, value=-0.25 + 0.0j)

        scalar = rho(K=0, Q=0, J=1, Jʹ=1, term_id=term_id)

        # The engine passes each argument as an [N, 1] column; the second row is an unset coherence.
        vector = rho.get_vector(
            term_id=np.array([[term_id], [term_id]]),
            K=np.array([[0.0], [9.0]]),
            Q=np.array([[0.0], [0.0]]),
            J=np.array([[1.0], [1.0]]),
            Jʹ=np.array([[1.0], [1.0]]),
        )

        assert vector.shape == (2,)
        assert np.isclose(vector[0], scalar)
        assert np.isnan(vector[1])


if __name__ == "__main__":
    unittest.main()
