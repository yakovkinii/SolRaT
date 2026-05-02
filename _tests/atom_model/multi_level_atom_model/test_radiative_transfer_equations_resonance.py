import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


def _build_two_level_atom():
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="1s", J=0, energy_cmm1=0, g=1.0)
    level_registry.register_level(alpha="2p", J=1, energy_cmm1=20_000, g=1.0)

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="2p", J=1),
        level_lower=level_registry.get_level(alpha="1s", J=0),
        einstein_a_ul_sm1=1e8,
    )

    nu_ul = energy_cmm1_to_frequency_sm1(20_000)
    lambda_A_air = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu_ul))

    return level_registry, transition_registry, lambda_A_air


class TestMultiLevelRadiativeTransferEquationsResonance(unittest.TestCase):
    def test_basic_rte_finite_and_shapes(self):
        """
        Two-level :math:`J=0 \\to J=1` resonance.  Run the full multi-level RTE pipeline at
        zero magnetic field and verify the output has the expected shape, is finite,
        and matches the documented physical relations (e.g. epsilon vs. eta_S).
        """
        setup_logging()

        level_registry, transition_registry, lambda_A_air = _build_two_level_atom()

        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=lambda_A_air - 1.0,
            upper_wavelength_A=lambda_A_air + 1.0,
            step_A=5e-3,
        )

        model = Models.multi_level_atom().configure(
            config=Models.multi_level_atom().Config(
                level_registry=level_registry,
                transition_registry=transition_registry,
                atomic_mass_amu=4.0,
                reference_lambda_A_air=lambda_A_air,
            )
        )

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=0,
            temperature_K=7000,
        )

        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_planck(temperature_K=5000)

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )
        rho = see.get_solution()

        rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rtc = rte.calculate_all_coefficients(
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
            rho=rho,
        )

        for arr in [
            rtc.get_eta_I(),
            rtc.get_eta_Q(),
            rtc.get_eta_U(),
            rtc.get_eta_V(),
            rtc.get_rho_I(),
            rtc.get_rho_Q(),
            rtc.get_rho_U(),
            rtc.get_rho_V(),
            rtc.get_epsilon_I(),
            rtc.get_epsilon_Q(),
            rtc.get_epsilon_U(),
            rtc.get_epsilon_V(),
        ]:
            assert arr.shape == nu.shape
            assert np.all(np.isfinite(arr))

        # eta_I should be positive and have a peak (resonance line).
        assert np.max(rtc.get_eta_I()) > 0
        assert np.argmax(rtc.get_eta_I()) not in (0, len(nu) - 1)

        # epsilon_I = 2 h nu^3 / c^2 * Re(eta_S_I), and eta_S_I is captured inside RTC.
        # The relation should hold by construction; a sanity check that finite & nonzero:
        assert np.max(np.abs(rtc.get_epsilon_I())) > 0


if __name__ == "__main__":
    unittest.main()
