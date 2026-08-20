import os
import tempfile
import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.nlte_state import NLTEState
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import frequencies_around_line_sm1
from solrat.atom_model.shared.utility.log_setup import setup_logging

TEMPERATURE_K = 6000.0


def log_depth_grid(z_max_cm, n_depth):
    r"""
    Height grid with the depth below the observer surface logarithmically spaced.
    """
    depth = np.logspace(np.log10(z_max_cm * 1e-6), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth)


def model_and_frequencies():
    r"""
    The mock two-level atom with a photon-destruction probability, and a coarse frequency grid.
    """
    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition, 1e-2, TEMPERATURE_K)
    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(),
        params.delta_v_thermal_cm_sm1,
        half_width_doppler=2.0,
        step_doppler=1.0,
    )
    return model, nu


def build_atmosphere(model, n_depth):
    r"""
    Tiny fixed-iteration self-consistent atmosphere on a grid of ``n_depth`` depths.
    """
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=log_depth_grid(1000e5, n_depth),
        temperature_K=TEMPERATURE_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    return NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=0.0,
        n_mu_quadrature=2,
        n_phi_quadrature=3,
        max_iterations=3,
        tolerance=1e-14,
        ng_acceleration=False,
    )


class TestNLTEState(unittest.TestCase):
    r"""
    Capture, save/load, interpolation, the compatibility guard, and warm-starting a second solve.
    """

    def setUp(self):
        setup_logging()
        self.model, self.nu = model_and_frequencies()
        self.atmosphere = build_atmosphere(self.model, n_depth=6)
        self.atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=self.nu))
        self.state = self.atmosphere.get_state()

    def test_capture_shapes(self):
        self.assertEqual(self.state.n_depth, 6)
        self.assertEqual(self.state.values.shape, (6, len(self.state.coherence_keys)))
        self.assertEqual(len(self.state.to_dicts()), 6)

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.npz")
            self.state.save(path)
            loaded = NLTEState.load(path)
        self.assertEqual(loaded.coherence_keys, self.state.coherence_keys)
        self.assertTrue(np.allclose(loaded.height_cm, self.state.height_cm))
        self.assertTrue(np.allclose(loaded.values, self.state.values))
        self.assertEqual(loaded.model_signature, self.state.model_signature)

    def test_interpolate_identity_and_resample(self):
        same = self.state.interpolate_to(self.state.height_cm)
        self.assertIs(same, self.state)  # identical grid returns self
        finer = self.state.interpolate_to(log_depth_grid(1000e5, 9))
        self.assertEqual(finer.n_depth, 9)
        self.assertEqual(finer.coherence_keys, self.state.coherence_keys)

    def test_check_compatible(self):
        self.state.check_compatible(self.state.model_signature, coherence_keys=self.state.coherence_keys)  # ok
        with self.assertRaises(AssertionError):
            self.state.check_compatible("other-model", coherence_keys=["does-not-exist"])  # no key overlap

    def test_warm_start_on_different_grid(self):
        warm = build_atmosphere(self.model, n_depth=8)  # different grid -> interpolate_to + apply_to_templates
        emergent = warm.forward(initial_stokes=Stokes.from_zeros(nu_sm1=self.nu), initial_state=self.state)
        for stokes in ("I", "Q", "U", "V"):
            self.assertTrue(np.all(np.isfinite(getattr(emergent, stokes))))
        self.assertEqual(warm.get_state().n_depth, 8)


if __name__ == "__main__":
    unittest.main()
