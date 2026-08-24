import logging

import matplotlib.pyplot as plt
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

try:
    from _demos.general.state.warm_start import load_warm_state, save_warm_state
except ImportError:
    from _demos.general.state.warm_start import load_warm_state, save_warm_state

WARM_START = True
WARM_START_ITERATIONS = 2


def log_depth_grid(z_max_cm: float, n_depth: int, min_fraction: float = 1e-9) -> np.ndarray:
    r"""
    Height grid with the depth below the observer surface logarithmically spaced. ``z[0]`` is the
    lower boundary (deep), ``z[-1]`` the observer surface (optical depth :math:`\to 0`).
    """
    depth_below_surface = np.logspace(np.log10(z_max_cm * min_fraction), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth_below_surface)


def upper_level_alignment(atmosphere: NLTEStratifiedAtmosphere, upper_level_id: str) -> np.ndarray:
    r"""
    Fractional atomic alignment :math:`\rho^2_0 / \rho^0_0` of the upper level over the depth grid
    (TB1999 Fig. 1).
    """
    sigma = []
    for rho in atmosphere.rho_grid:
        rho00 = np.real(rho(K=0, Q=0, level_id=upper_level_id))
        rho20 = np.real(rho(K=2, Q=0, level_id=upper_level_id))
        sigma.append(rho20 / rho00)
    return np.array(sigma)


def main():
    r"""
    Reproduce the Trujillo Bueno & Manso Sainz (1999), ApJ 516, 436, resonance-line-polarization
    benchmark: a :math:`J=0 \to 1` two-level atom in an isothermal, self-emitting, plane-parallel
    slab, with the photon destruction probability set through the parametrized collisional
    de-excitation rate.

    Plots the upper-level alignment :math:`\rho^2_0/\rho^0_0` versus optical depth from the surface
    against the tabulated surface value, and logs the tangential (:math:`\mu=0`) line-center Q/I
    against TB1999 Table 4.
    """
    setup_logging()

    temperature_K = 6000.0
    epsilon = 1.0e-2
    mu_observer = 0.0
    tb1999_surface_alignment = 0.05666  # TB1999 Table 4
    tb1999_qi_percent_tangential = -6.132  # TB1999 Table 4

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    upper_level_id = transition.level_upper.level_id
    collisions.set_deexcitation_rate_from_epsilon(transition, epsilon, temperature_K)

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(), params.delta_v_thermal_cm_sm1, step_doppler=0.5
    )
    line_center_index = int(np.argmin(np.abs(nu - transition.get_mean_transition_frequency_sm1())))

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=log_depth_grid(1000e5, 80),
        temperature_K=temperature_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    initial_state = load_warm_state(__file__, WARM_START)
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(mu_observer)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=WARM_START_ITERATIONS if initial_state is not None else 1000,
        tolerance=1e-8,
        ng_acceleration=True,
        ng_damping=0.7,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu), initial_state=initial_state)
    save_warm_state(__file__, atmosphere)

    vertical_tau = atmosphere.tau_grid
    optical_depth_from_surface = vertical_tau[-1] - vertical_tau
    alignment = upper_level_alignment(atmosphere, upper_level_id)
    surface_alignment = alignment[-1]
    emergent_qi_percent = 100.0 * emergent.Q[line_center_index] / emergent.I[line_center_index]

    logging.info("TB1999 benchmark: epsilon = %.0e, delta2 = 0, no continuum, no field", epsilon)
    logging.info(
        "vertical optical thickness = %.1f, iterations = %d, residual = %.2e",
        float(vertical_tau[-1]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )
    logging.info("surface rho^2_0/rho^0_0 = %.5f  (TB1999: %.5f)", surface_alignment, tb1999_surface_alignment)
    logging.info(
        "tangential (mu=0) line-center Q/I = %.3f %%  (TB1999 Table 4: %.3f %%)",
        emergent_qi_percent,
        tb1999_qi_percent_tangential,
    )

    fig_alignment, ax_alignment = plt.subplots(figsize=(7, 5))
    ax_alignment.axhline(
        tb1999_surface_alignment,
        color="k",
        linestyle="--",
        label=f"TB1999 surface value = {tb1999_surface_alignment}",
    )
    ax_alignment.plot(optical_depth_from_surface[:-1], alignment[:-1], marker=".", label="SolRaT")
    ax_alignment.set_xscale("log")
    ax_alignment.set_xlabel(r"optical depth from surface  $\tau$")
    ax_alignment.set_ylabel(r"upper-level alignment  $\rho^2_0 / \rho^0_0$")
    ax_alignment.set_ylim(-0.02, 0.10)
    ax_alignment.legend()
    fig_alignment.tight_layout()

    print(
        f"TB1999 (epsilon={epsilon:.0e}): surface rho^2_0/rho^0_0 = {surface_alignment:.5f} "
        f"(TB1999 {tb1999_surface_alignment:.5f}, rel err "
        f"{abs(surface_alignment / tb1999_surface_alignment - 1.0):.1%}); tangential Q/I = "
        f"{emergent_qi_percent:.3f}% (TB1999 {tb1999_qi_percent_tangential:.3f}%); "
        f"iterations = {atmosphere.iterations_used}"
    )
    return fig_alignment


if __name__ == "__main__":
    main()
    plt.show()
