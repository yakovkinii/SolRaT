import logging
import unittest

import numpy as np

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.utility.paschen_back import _g_ls, calculate_paschen_back
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestArtificialSpinScaleLande(unittest.TestCase):
    r"""
    The experimental ``artificial_S_scale`` override of the LS Lande factor.
    """

    def test_zero_g_for_J0(self):
        assert _g_ls(L=1, S=1, J=0) == 0

    def test_artificial_scale_rescales_the_anomalous_part(self):
        # g = 1 + scale * [J(J+1)+S(S+1)-L(L+1)] / (2 J (J+1)); for ^3P_2 the bracket / (2 J (J+1)) = 1/2.
        assert abs(_g_ls(L=1, S=1, J=2, artificial_S_scale=2.0) - 2.0) < 1e-12


class TestPaschenBackWithArtificialScale(unittest.TestCase):
    r"""
    The incomplete-Paschen-Back diagonalization with an artificial spin scale set on the term
    (drives the scaled diagonal and off-diagonal Hamiltonian entries).
    """

    def test_diagonalization_runs_with_artificial_scale(self):
        setup_logging()
        registry = LevelRegistry()
        for J, energy in [(0, 100.0), (1, 101.0), (2, 103.0)]:
            registry.register_level(beta="3P", L=1, S=1, J=J, energy_cmm1=energy)
        registry.validate()
        term = registry.get_term(beta="3P", L=1, S=1)
        term.set_artificial_spin_scale(2.0)

        eigenvalues, coefficients = calculate_paschen_back(term, 1000.0)
        assert len(eigenvalues.data) > 0
        assert len(coefficients.data) > 0

        # Order-independent regression fingerprint of the eigenvalues and coefficients.
        eig = np.array(sorted(eigenvalues.data.values()))
        coeff = np.array(sorted(coefficients.data.values()))
        fingerprint = float(np.sum(eig * np.arange(1, eig.size + 1)) + np.sum(coeff * np.arange(1, coeff.size + 1)))
        previous_fingerprint = 4751.122479260735
        logging.info(
            f"test_diagonalization_runs_with_artificial_scale current={fingerprint!r} previous={previous_fingerprint!r}"
        )
        assert np.isfinite(fingerprint)
        assert np.isclose(fingerprint, previous_fingerprint, rtol=1e-8, atol=0.0)


if __name__ == "__main__":
    unittest.main()
