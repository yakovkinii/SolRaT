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
    return get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.3,
        upper_wavelength_A=reference_lambda_A_air + 0.3,
        step_A=5e-2,
    )


class TestStratifiedAtmosphereFeatures(unittest.TestCase):
    r"""
    Optional code paths of the iterated stratified atmosphere: the velocity-under-resolution warning,
    the per-iteration callback, and a tangential observer ray.
    """

    def test_velocity_gradient_underresolved_warns(self):
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)
        # Coarse grid with a steep vertical velocity: the per-cell shift far exceeds the thermal width.
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=np.linspace(0.0, 150e5, 3),
            temperature_K=6000.0,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            velocity_cm_sm1=lambda z: 4.0e6 * (z / 150e5),  # 0 -> 40 km/s
            theta_v=0.0,
            delta_v_turbulent_cm_sm1=0.0,
            continuum_to_line_ratio=1e-2,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.2,
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            max_iterations=1,
            tolerance=1.0,
        )
        with self.assertLogs(level="WARNING") as captured:
            atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700))
        assert any("velocity" in message.lower() for message in captured.output)

    def test_on_iteration_callback_is_invoked(self):
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=np.linspace(0.0, 150e5, 3),
            temperature_K=6000.0,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_to_line_ratio=1e-2,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.3,
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            max_iterations=2,
            tolerance=1e-9,  # force both iterations so the callback fires more than once
        )
        recorded = []
        atmosphere.forward(
            initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700),
            on_iteration=lambda iteration, emergent: recorded.append((iteration, emergent)),
        )
        assert len(recorded) >= 1
        assert isinstance(recorded[0][1], Stokes)

    def test_tangential_observer_ray(self):
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=np.linspace(0.0, 150e5, 3),
            temperature_K=6000.0,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=50.0,
            theta_B=0.3,
            delta_v_turbulent_cm_sm1=2.0e5,
            continuum_to_line_ratio=1e-2,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=np.pi / 2,  # tangential: |mu| below the threshold
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            max_iterations=1,
            tolerance=1.0,
        )
        emergent = atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700))
        new_hash = pseudo_hash(emergent.I, emergent.Q, emergent.U, emergent.V)
        previous_hash = 7.726856867081245e-05
        logging.info(f"test_tangential_observer_ray_runs current={new_hash!r} previous={previous_hash!r}")
        assert np.isclose(new_hash, previous_hash, rtol=1e-8, atol=0.0)


if __name__ == "__main__":
    unittest.main()
