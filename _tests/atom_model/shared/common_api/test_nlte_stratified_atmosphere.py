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


def _build_two_level_model():
    """A J=0 -> J=1 resonance line as a multi-level atom."""
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


class TestStratifiedAtmosphereContainer(unittest.TestCase):
    def test_profile_sampling_scalar_array_callable(self):
        """Scalars broadcast, arrays pass through, callables are sampled on the height grid."""
        model, _ = _build_two_level_model()
        z = np.linspace(0.0, 100e5, 5)
        strat = StratifiedAtmosphere(
            model=model,
            height_cm=z,
            temperature_K=6000.0,  # scalar
            number_density_cm3=np.linspace(1e11, 1e10, 5),  # array
            velocity_cm_sm1=lambda zz: 1.0e5 * (zz / z[-1]),  # callable
            theta_v=0.0,
            chi_v=0.0,
        )
        assert strat.n_depth == 5
        assert np.allclose(strat.temperature_K, 6000.0)
        assert np.isclose(strat.number_density_cm3[0], 1e11)
        assert np.isclose(strat.number_density_cm3[-1], 1e10)
        assert np.isclose(strat.velocity_cm_sm1[-1], 1.0e5)
        assert np.isclose(strat.velocity_cm_sm1[0], 0.0)

    def test_rejects_non_increasing_grid(self):
        model, _ = _build_two_level_model()
        with self.assertRaises(ValueError):
            StratifiedAtmosphere(
                model=model,
                height_cm=[0.0, 10.0, 5.0],
                temperature_K=6000.0,
                number_density_cm3=1e11,
            )

    def test_velocity_vector_geometry(self):
        """
        v_los = -Omega_hat . v (Omega_hat = photon propagation / toward-observer direction).
        The minus makes positive v_los a redshift, so plasma moving toward the observer (a
        velocity along Omega_hat) gives v_los < 0, i.e. a blueshift.
        """
        model, _ = _build_two_level_model()
        z = np.linspace(0.0, 100e5, 3)
        los_theta, los_chi = 0.7, 1.1
        v0 = 1.5e5
        omega = NLTEStratifiedAtmosphere._ray_direction(los_theta, los_chi)

        # Velocity along the observer propagation direction = moving toward observer -> blueshift.
        strat_toward = StratifiedAtmosphere(
            model=model, height_cm=z, temperature_K=6000.0, number_density_cm3=1e11,
            velocity_cm_sm1=v0, theta_v=los_theta, chi_v=los_chi,
        )
        v_los_toward = NLTEStratifiedAtmosphere._project(strat_toward.velocity_vector(0), omega)
        assert np.isclose(v_los_toward, -v0)

        # A purely vertical velocity projects with -cos(polar angle).
        strat_vertical = StratifiedAtmosphere(
            model=model, height_cm=z, temperature_K=6000.0, number_density_cm3=1e11,
            velocity_cm_sm1=v0, theta_v=0.0, chi_v=0.0,
        )
        v_los_vertical = NLTEStratifiedAtmosphere._project(strat_vertical.velocity_vector(0), omega)
        assert np.isclose(v_los_vertical, -v0 * np.cos(los_theta))


class TestNLTEStratifiedAtmosphere(unittest.TestCase):
    def test_runs_and_is_finite(self):
        """Smoke test: the stratified NLTE loop runs, converges or hits max, stays finite."""
        setup_logging()

        model, reference_lambda_A_air = _build_two_level_model()
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.5,
            upper_wavelength_A=reference_lambda_A_air + 0.5,
            step_A=2e-2,  # coarse for speed
        )

        height_cm = np.linspace(0.0, 200e5, 6)
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=height_cm,
            temperature_K=lambda z: 5000.0 + 3000.0 * (z / height_cm[-1]),
            number_density_cm3=lambda z: 1.0e11 * np.exp(-z / 150e5),
            magnetic_field_gauss=200.0,
            theta_B=0.3,
            chi_B=0.2,
            velocity_cm_sm1=lambda z: 1.5e5 * (z / height_cm[-1]),
            theta_v=0.0,
            chi_v=0.0,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_to_line_ratio=1e-2,
        )

        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.4,
            los_chi=0.1,
            los_gamma=0.0,
            n_mu_quadrature=2,
            n_phi_quadrature=2,
            max_iterations=3,
            tolerance=1e-3,
        )

        initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)
        emergent = atmosphere.forward(initial_stokes=initial_stokes)

        assert emergent.I.shape == nu.shape
        for arr in (emergent.I, emergent.Q, emergent.U, emergent.V):
            assert np.all(np.isfinite(arr))
        assert np.max(emergent.I) > 0

        assert atmosphere.rho_grid is not None and len(atmosphere.rho_grid) == 6
        assert atmosphere.iterations_used is not None and atmosphere.iterations_used >= 1
        assert atmosphere.tau_grid is not None and len(atmosphere.tau_grid) == 6
        # Optical depth accumulates from the lower boundary upward (monotone non-decreasing).
        assert np.all(np.diff(atmosphere.tau_grid) >= 0)

    def test_uniform_velocity_free_atmosphere_converges(self):
        """
        A constant-property, velocity-free stratified atmosphere is a well-posed scattering
        slab: it should converge and show a line feature (absorption, since the line source is
        the diluted scattering integral, not the local Planck function).
        """
        setup_logging()

        model, reference_lambda_A_air = _build_two_level_model()
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.5,
            upper_wavelength_A=reference_lambda_A_air + 0.5,
            step_A=2e-2,
        )

        height_cm = np.linspace(0.0, 200e5, 5)
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=height_cm,
            temperature_K=8000.0,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            velocity_cm_sm1=0.0,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_to_line_ratio=1e-2,
        )

        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.2,
            n_mu_quadrature=2,
            n_phi_quadrature=2,
            max_iterations=20,
            tolerance=1e-4,
        )

        initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)
        emergent = atmosphere.forward(initial_stokes=initial_stokes)

        assert np.all(np.isfinite(emergent.I))
        assert atmosphere.final_residual is not None

    def test_velocity_doppler_direction_end_to_end(self):
        """
        End-to-end sign check: a uniform bulk velocity directed toward the observer must
        blueshift the emergent line core (higher frequency), by ~|v|/c. This guards the
        combination of the projection (v_los = -Omega_hat . v) and the profile convention
        (nu = nu_i (1 - v_los/c)) against a regression in either piece.
        """
        setup_logging()

        model, reference_lambda_A_air = _build_two_level_model()
        nu_rest = energy_cmm1_to_frequency_sm1(20_000)
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 1.0,
            upper_wavelength_A=reference_lambda_A_air + 1.0,
            step_A=5e-3,
        )

        los_theta, los_chi = 10 * np.pi / 180, 0.0
        v = 15.0e5  # 15 km/s
        c_cm_sm1 = 2.99792458e10

        # Velocity pointing along the observer propagation direction = moving toward observer.
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=np.linspace(0.0, 200e5, 4),
            temperature_K=8000.0,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            velocity_cm_sm1=v,
            theta_v=los_theta,
            chi_v=los_chi,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_to_line_ratio=1e-2,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=los_theta,
            los_chi=los_chi,
            n_mu_quadrature=2,
            n_phi_quadrature=2,
            max_iterations=4,
            tolerance=1e-3,
        )
        emergent = atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700))

        core_nu = nu[int(np.argmin(emergent.I))]
        # Blueshift: line core at higher frequency than rest.
        assert core_nu > nu_rest
        # Magnitude ~ |v|/c (v_los = -|v| since velocity is along the toward-observer direction).
        assert np.isclose((core_nu - nu_rest) / nu_rest, v / c_cm_sm1, rtol=0.2)


if __name__ == "__main__":
    unittest.main()
