import unittest

import numpy as np

from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.radiation_tensor import RadiationTensor
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging


def build_config():
    r"""
    A single :math:`J=0 \to 1` transition near 5000 Angstrom (in the n/w-fit validity range).
    """
    levels = LevelRegistry()
    levels.register_level(alpha="lower", J=0, energy_cmm1=0.0, g=1.0)
    levels.register_level(alpha="upper", J=1, energy_cmm1=20000.0, g=1.0)
    transitions = TransitionRegistry()
    transitions.register_transition(
        level_upper=levels.get_level(alpha="upper", J=1),
        level_lower=levels.get_level(alpha="lower", J=0),
        einstein_a_ul_sm1=1.0e7,
    )
    return MultiLevelAtomConfig(
        level_registry=levels,
        transition_registry=transitions,
        atomic_mass_amu=40.0,
        reference_lambda_A_air=5000.0,
        collisions=None,
    )


class TestMultiLevelRadiationTensor(unittest.TestCase):
    r"""
    The multi-level J^K_Q container: Planck and anisotropic fills, the dataframe view, and the identity
    magnetic-frame rotation.
    """

    def setUp(self):
        setup_logging()
        self.config = build_config()
        self.transition_id = next(iter(self.config.transition_registry.transitions.values())).transition_id

    def test_planck_is_isotropic(self):
        tensor = RadiationTensor.from_model_config(self.config).fill_planck(temperature_K=6000.0)
        self.assertGreater(np.real(tensor.get_from_transition_id(self.transition_id, K=0, Q=0)), 0.0)
        self.assertAlmostEqual(np.real(tensor.get_from_transition_id(self.transition_id, K=2, Q=0)), 0.0)
        self.assertGreater(len(tensor.df), 0)  # triggers construct_df

    def test_anisotropic_has_alignment(self):
        tensor = RadiationTensor.from_model_config(self.config).fill_NLTE_n_w_allen(h_arcsec=30)
        self.assertGreater(np.real(tensor.get_from_transition_id(self.transition_id, K=0, Q=0)), 0.0)
        self.assertNotAlmostEqual(np.real(tensor.get_from_transition_id(self.transition_id, K=2, Q=0)), 0.0)
        self.assertGreater(len(tensor.df), 0)

    def test_isotropic_rotation_is_identity(self):
        tensor = RadiationTensor.from_model_config(self.config).fill_planck(temperature_K=6000.0)
        rotated = tensor.rotate_to_magnetic_frame(angles=Angles())
        before = np.real(tensor.get_from_transition_id(self.transition_id, K=0, Q=0))
        after = np.real(rotated.get_from_transition_id(self.transition_id, K=0, Q=0))
        self.assertAlmostEqual(before, after)


if __name__ == "__main__":
    unittest.main()
