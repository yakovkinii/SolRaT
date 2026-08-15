import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sqrt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, kB_erg_Km1
from solrat.atom_model.shared.utility.log_setup import setup_logging

_LOCAL_TEMPERATURE_K = 6000.0


def population_ratio(model, transition) -> float:
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


def boltzmann_population_ratio(transition) -> float:
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


def main():
    r"""
    Demo: the LTE (Boltzmann) limit of the parametrized collisions, as an epsilon-bridge.

    A two-level mock atom is solved for a sweep of collisional de-excitation rates C_ul. As collisions
    grow large compared with the radiative rates (A_ul = 1e7), the upper/lower population ratio n_u/n_l
    is dragged away from the collisionless (pure-scattering) value and onto the Boltzmann distribution at
    the local temperature (LL04 eq. 7.98, detailed balance) - the horizontal reference line - even though
    the radiation temperature is different. This is the LTE <-> NLTE bridge that the collisions provide.
    """
    setup_logging()

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))

    boltzmann_ratio = boltzmann_population_ratio(transition)
    c_ul_sweep = np.logspace(5, 13, 40)  # 1e5 (scattering) .. 1e13 (collision-dominated)
    ratios = np.empty_like(c_ul_sweep)
    for index, c_ul in enumerate(c_ul_sweep):
        collisions.set_deexcitation_rate(transition.transition_id, float(c_ul))
        ratios[index] = population_ratio(model, transition)

    logging.info("Boltzmann (LTE) ratio at %.0f K: %.6e", _LOCAL_TEMPERATURE_K, boltzmann_ratio)
    logging.info("strong-collision ratio (C_ul = %.0e): %.6e", c_ul_sweep[-1], ratios[-1])

    plt.figure(figsize=(7, 5))
    plt.axhline(
        boltzmann_ratio,
        color="k",
        linestyle="--",
        label=f"Boltzmann / LTE at {_LOCAL_TEMPERATURE_K:.0f} K",
    )
    plt.plot(c_ul_sweep, ratios, marker="o", markersize=3, label="SEE population ratio")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"collisional de-excitation rate $C_{ul}$  [s$^{-1}$]")
    plt.ylabel(r"population ratio  $n_u / n_l$")
    plt.title("Collisional LTE limit (bridge to Boltzmann)")
    plt.legend()
    plt.tight_layout()
    print(
        f"Collisionless-to-LTE limit: strong-collision n_u/n_l vs Boltzmann relative error = "
        f"{abs(ratios[-1] / boltzmann_ratio - 1.0):.2e} (C_ul = {c_ul_sweep[-1]:.0e})"
    )
    plt.show()


if __name__ == "__main__":
    main()
