import unittest

import numpy as np
from numpy import exp, sqrt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, kB_erg_Km1
from solrat.atom_model.shared.utility.log_setup import setup_logging

_LOCAL_TEMPERATURE_K = 6000.0


def _population_ratio(model, transition) -> float:
    r"""
    Solve the SEE and return the upper/lower population ratio n_u/n_l, using the LL04 relation between
    the total population of a level and its rank-0 tensor: N(alpha J) = sqrt(2J + 1) rho^0_0(alpha J).
    """
    lower = transition.level_lower
    upper = transition.level_upper
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=_LOCAL_TEMPERATURE_K
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_planck(temperature_K=8000.0)
    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=Angles()),
    )
    rho = see.get_solution()
    population_lower = sqrt(2 * lower.J + 1) * np.real(rho(K=0, Q=0, level_id=lower.level_id))
    population_upper = sqrt(2 * upper.J + 1) * np.real(rho(K=0, Q=0, level_id=upper.level_id))
    return population_upper / population_lower


def _boltzmann_population_ratio(transition) -> float:
    r"""
    The LTE (Boltzmann) upper/lower population ratio at the local temperature:
    n_u/n_l = (2J_u + 1)/(2J_l + 1) exp(-(E_u - E_l)/(k_B T)). This is exactly the ratio the coded
    detailed-balance relation c_lu/c_ul (LL04 eq. 7.98) collapses to in the collision-dominated limit.
    """
    lower = transition.level_lower
    upper = transition.level_upper
    delta_e_erg = (upper.energy_cmm1 - lower.energy_cmm1) * h_erg_s * c_cm_sm1
    degeneracy_ratio = (2 * upper.J + 1) / (2 * lower.J + 1)
    return degeneracy_ratio * exp(-delta_e_erg / (kB_erg_Km1 * _LOCAL_TEMPERATURE_K))


class TestCollisionsLTELimit(unittest.TestCase):
    r"""
    LTE-limit (epsilon-bridge) validation of the parametrized collisions: as collisional de-excitation
    grows large compared with the radiative rates, the multi-level SEE populations must relax to the
    Boltzmann distribution at the local temperature (LL04 eq. 7.98, detailed balance).
    """

    def test_strong_collisions_reproduce_boltzmann(self):
        r"""
        With collisional de-excitation dominating the radiative rates (C_ul >> A_ul, B J), the upper/lower
        population ratio matches the Boltzmann value at the local temperature to high precision.
        """
        setup_logging()
        collisions = ParametrizedCollisions()
        model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
        transition = next(iter(model.config.transition_registry.transitions.values()))
        collisions.set_deexcitation_rate(transition.transition_id, 1.0e12)  # C_ul >> A_ul = 1e7

        ratio = _population_ratio(model, transition)
        assert np.isclose(ratio, _boltzmann_population_ratio(transition), rtol=1e-3)

    def test_bridge_from_scattering_to_lte(self):
        r"""
        The epsilon-bridge: increasing the collisional rate moves the population ratio from the
        collisionless (pure-scattering) value toward the Boltzmann/LTE value. The strong-collision
        solution is strictly closer to Boltzmann than the weak-collision solution.
        """
        setup_logging()
        collisions = ParametrizedCollisions()
        model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
        transition = next(iter(model.config.transition_registry.transitions.values()))
        boltzmann_ratio = _boltzmann_population_ratio(transition)

        collisions.set_deexcitation_rate(transition.transition_id, 1.0e6)  # comparable to / below A_ul
        ratio_weak = _population_ratio(model, transition)
        collisions.set_deexcitation_rate(transition.transition_id, 1.0e12)  # collisions dominate
        ratio_strong = _population_ratio(model, transition)

        assert abs(ratio_strong - boltzmann_ratio) < abs(ratio_weak - boltzmann_ratio)
        assert np.isclose(ratio_strong, boltzmann_ratio, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
