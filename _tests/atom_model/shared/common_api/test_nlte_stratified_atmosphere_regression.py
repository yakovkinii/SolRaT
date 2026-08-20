import logging
import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.special import pseudo_hash


def build_two_level_model():
    r"""
    A J=0 -> J=1 resonance line as a multi-level atom.
    """
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="1s", J=0, energy_cmm1=0, g=1.0)
    level_registry.register_level(alpha="2p", J=1, energy_cmm1=20_000, g=1.2)

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="2p", J=1),
        level_lower=level_registry.get_level(alpha="1s", J=0),
        einstein_a_ul_sm1=1e7,
    )

    nu_ul = energy_cmm1_to_frequency_sm1(20_000)
    reference_lambda_A_air = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu_ul))

    model = Models.multi_level_atom()
    model = model.configure(
        config=model.Config(
            level_registry=level_registry,
            transition_registry=transition_registry,
            atomic_mass_amu=4.0,
            reference_lambda_A_air=reference_lambda_A_air,
        )
    )
    return model, reference_lambda_A_air


def tiny_nu(reference_lambda_A_air):
    r"""
    Coarse frequency grid for a fast run.
    """
    return get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.4,
        upper_wavelength_A=reference_lambda_A_air + 0.4,
        step_A=5e-2,
    )


class TestStratifiedAtmosphereContainer(unittest.TestCase):
    r"""
    Sampling of scalar/array/callable profiles onto the depth grid and the per-depth getters.
    """

    def test_construction_sampling_and_getters(self):
        r"""
        Scalar, array, and callable profiles are sampled onto the grid, and the per-depth getters
        return the expected shapes/types.
        """
        setup_logging()
        model = build_two_level_model()[0]
        z = np.linspace(0.0, 100e5, 4)

        strat = StratifiedAtmosphere(
            model=model,
            height_cm=z,
            temperature_K=6000.0,  # scalar
            number_density_cm3=np.linspace(1e11, 1e10, 4),  # array
            magnetic_field_gauss=lambda zz: 100.0 + zz / 1e5,  # callable
            velocity_cm_sm1=1.0e5,
            theta_v=0.3,
            chi_v=0.2,
        )
        assert strat.n_depth == 4
        assert np.allclose(strat.temperature_K, 6000.0)
        assert strat.number_density_cm3.shape == z.shape
        assert strat.velocity_vector(0).shape == (3,)
        assert strat.magnetic_frame_angles(1).theta_B == strat.theta_B[1]
        params = strat.atmosphere_parameters(2, macroscopic_velocity_cm_sm1=1234.0)
        assert params.macroscopic_velocity_cm_sm1 == 1234.0

        uniform = StratifiedAtmosphere.on_uniform_grid(
            model=model, z_min_cm=0.0, z_max_cm=50e5, n_depth=3, temperature_K=5000.0, number_density_cm3=1e11
        )
        assert uniform.n_depth == 3

    def test_rejects_invalid_input(self):
        r"""
        Invalid user input trips the assertions.
        """
        model = build_two_level_model()[0]
        with self.assertRaises(AssertionError):
            StratifiedAtmosphere(model=model, height_cm=[0.0, 10.0, 5.0], temperature_K=6000.0, number_density_cm3=1e11)
        with self.assertRaises(AssertionError):
            StratifiedAtmosphere(model=model, height_cm=[0.0, 10.0], temperature_K=-1.0, number_density_cm3=1e11)


class TestStratifiedNLTEAtmosphereForwardRegression(unittest.TestCase):
    r"""
    A tiny self-consistent run reduced to a ``pseudo_hash`` of its emergent Stokes vector, locked
    against a baseline. Each test logs the computed hash; on the first run (or after an intended
    numerical change) copy the logged value into that test's ``last_run_hash``.
    """

    def test_forward_ratio_continuum(self):
        r"""
        Regression: tiny self-consistent run with the continuum-to-line ratio path.
        """
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)

        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=np.linspace(0.0, 150e5, 3),
            temperature_K=lambda z: 6000.0 + 2000.0 * (z / 150e5),
            number_density_cm3=1.0e11,
            magnetic_field_gauss=150.0,
            theta_B=0.3,
            chi_B=0.2,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_to_line_ratio=1e-2,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.4,
            los_chi=0.1,
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            max_iterations=2,
            tolerance=1e-3,
        )
        emergent = atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700))
        assert atmosphere.tau_grid is not None and np.all(np.diff(atmosphere.tau_grid) >= 0)

        new_hash = pseudo_hash(emergent.I, emergent.Q, emergent.U, emergent.V)
        logging.info(f"\ntest_forward_ratio_continuum pseudo_hash = {new_hash!r}")
        last_run_hash = 6.98600589400781e-05
        assert np.isfinite(new_hash)
        assert np.abs((last_run_hash - new_hash) / last_run_hash) < 1e-8

    def test_forward_explicit_continuum_and_velocity(self):
        r"""
        Regression: tiny run with an explicit continuum-opacity profile and a bulk velocity (exercises
        the explicit-k_c branch and the per-ray velocity projection).
        """
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)

        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=np.linspace(0.0, 150e5, 3),
            temperature_K=7000.0,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            velocity_cm_sm1=lambda z: 1.5e5 * (z / 150e5),
            theta_v=0.0,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_opacity_cm_m1=1e-10,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.3,
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            max_iterations=1,
            tolerance=1.0,
        )
        emergent = atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700))

        new_hash = pseudo_hash(emergent.I, emergent.Q, emergent.U, emergent.V)
        logging.info(f"\ntest_forward_explicit_continuum_and_velocity pseudo_hash = {new_hash!r}")
        last_run_hash = 7.82804574476679e-05
        assert np.isfinite(new_hash)
        assert np.abs((last_run_hash - new_hash) / last_run_hash) < 1e-8


if __name__ == "__main__":
    unittest.main()
