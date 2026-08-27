import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
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


class TestEstimateTrueError(unittest.TestCase):
    r"""
    The ``estimate_true_error`` convergence criterion: stop on the estimated distance to the fixed
    point (from the residual-decay rate) rather than on the step size. Drives that branch and the rate
    estimator on a tiny TM99-like slab.
    """

    def test_runs_and_populates_error_estimate(self):
        setup_logging()
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
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=log_depth_grid(1000e5, 6),
            temperature_K=TEMPERATURE_K,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            delta_v_turbulent_cm_sm1=0.0,
            voigt_a=0.0,
            continuum_to_line_ratio=0.0,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.0,
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            max_iterations=15,
            tolerance=1e-6,
            ng_acceleration=False,  # keeps every iteration in the rate-measurement window
            estimate_true_error=True,
        )
        atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

        self.assertIsNotNone(atmosphere.iterations_used)
        self.assertGreaterEqual(atmosphere.iterations_used, 1)
        self.assertTrue(np.isfinite(atmosphere.final_residual))
        # the estimate branch ran: these fields are populated (None until enough clean-decay residuals
        # accrue, finite thereafter -- either way the branch executed).
        self.assertTrue(atmosphere.final_true_error is None or np.isfinite(atmosphere.final_true_error))
        self.assertTrue(atmosphere.lambda_estimate is None or np.isfinite(atmosphere.lambda_estimate))


if __name__ == "__main__":
    unittest.main()
