import logging

import matplotlib.pyplot as plt
import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    frequencies_around_line_sm1,
    height_grid_refined_at_observer_surface,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging

try:
    from _demos.general.state.warm_start import load_warm_state, save_warm_state
except ImportError:
    from _demos.general.state.warm_start import load_warm_state, save_warm_state

WARM_START = True
WARM_START_ITERATIONS = 2


def slab_height_for_tau_total(model, temperature_K, number_density_cm3, nu, target_tau_total):
    r"""
    Slab height [cm] giving ``target_tau_total`` line-integrated optical thickness (:math:`\tau` is
    linear in the height at fixed number density, so one coarse probe fixes the scale).
    """
    probe_height_cm = 1.0e9
    probe = NLTEStratifiedAtmosphere(
        model=model,
        stratification=StratifiedAtmosphere(
            model=model,
            height_cm=height_grid_refined_at_observer_surface(probe_height_cm, n_near_surface=10, n_interior=5),
            temperature_K=temperature_K,
            number_density_cm3=number_density_cm3,
        ),
        los_theta=0.0,
        n_mu_quadrature=2,
        n_phi_quadrature=3,
        max_iterations=1,
        tolerance=1.0,
    )
    probe.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    return probe_height_cm * target_tau_total / float(probe.tau_grid[-1])


def line_center_source_function(atmosphere, nu, line_center):
    r"""
    Line-center source function :math:`S_I = \epsilon_I/\eta_I` per depth [erg ...] from the converged
    density-matrix grid (vertical ray; N cancels in the ratio).
    """
    model = atmosphere.model
    strat = atmosphere.stratification
    rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
    rte.N = 1.0
    vertical = Angles(chi=0.0, theta=0.0, gamma=0.0, chi_B=0.0, theta_B=0.0)
    source = np.empty(strat.n_depth)
    for i in range(strat.n_depth):
        rtc = rte.calculate_all_coefficients(
            atmosphere_parameters=strat.atmosphere_parameters(i, 0.0), angles=vertical, rho=atmosphere.rho_grid[i]
        )
        eta_I = float(np.real(rtc.get_eta_I()[line_center]))
        eps_I = float(np.real(rtc.epsilon_z()[line_center, 0, 0]))
        source[i] = eps_I / eta_I
    return source


def main():
    r"""
    Line source function :math:`S(\tau)/B` of a finite-:math:`\epsilon`, isothermal, semi-infinite
    two-level atom versus the :math:`\sqrt\epsilon` thermalization law: it rises from the surface value
    :math:`S(0)/B=\sqrt\epsilon` (dotted) to :math:`B` in the deep interior. This is a Doppler line in
    complete redistribution, so it thermalizes over the Doppler-line depth scale
    :math:`\tau\sim1/\epsilon` in line-center optical depth (photons escape through the low-opacity
    wings), not over the monochromatic :math:`1/\sqrt{3\epsilon}`; only the surface value and the deep
    limit are profile independent, and both are what the law fixes.

    :math:`B` is read as the thermalized interior value of :math:`S` itself (many thermalization
    depths below the observer surface), so the test is independent of the absolute intensity units.
    """
    setup_logging()

    temperature_K = 6000.0
    number_density_cm3 = 1.0e11
    epsilon_values = [1e-2, 1e-3]  # the second warm-starts from the first
    points_per_decade = 15

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu0 = transition.get_mean_transition_frequency_sm1()
    nu = frequencies_around_line_sm1(nu0, params.delta_v_thermal_cm_sm1, half_width_doppler=5.0, step_doppler=0.25)
    line_center = int(np.argmin(np.abs(nu - nu0)))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.7, len(epsilon_values)))

    warm_state = load_warm_state(__file__, WARM_START)
    state = warm_state
    for epsilon, color in zip(epsilon_values, colors):
        collisions.set_deexcitation_rate_from_epsilon(
            transition=transition, epsilon=epsilon, temperature_K=temperature_K
        )
        # Semi-infinite: total thickness >> the thermalization depth 1/epsilon, so the surface value
        # is the semi-infinite limit.
        slab_height_cm = slab_height_for_tau_total(
            model, temperature_K, number_density_cm3, nu, target_tau_total=100.0 / epsilon
        )
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=height_grid_refined_at_observer_surface(
                slab_height_cm, 4 * points_per_decade, 3 * points_per_decade
            ),
            temperature_K=temperature_K,
            number_density_cm3=number_density_cm3,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=0.0,
            n_mu_quadrature=8,
            n_phi_quadrature=3,
            max_iterations=WARM_START_ITERATIONS if warm_state is not None else 20000,
            tolerance=1e-12,
            ng_acceleration=True,
            ng_damping=0.5,
            ng_period=10,
            transfer_scheme="delo_linear",
        )
        # Thermalized (Planck) lower boundary: makes the slab semi-infinite, so only the observer
        # surface shows the sqrt-epsilon dip (a vacuum lower boundary would be a second free surface).
        atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=temperature_K), initial_state=state)
        state = atmosphere.get_state()

        # tau_grid is 0 at the lower boundary z[0] and tau_total at the observer surface z[-1]; the
        # sqrt-epsilon law is written in optical depth measured from that surface inward.
        tau_from_surface = float(atmosphere.tau_grid[-1]) - atmosphere.tau_grid
        source = line_center_source_function(atmosphere, nu, line_center)
        # Thermalized interior (deep, far below the observer surface) stands in for B, so the ratio is
        # unit-free; only the observer surface dips to sqrt(epsilon) B (the lower boundary is Planck).
        planck_plateau = float(source[np.argmin(np.abs(tau_from_surface - 0.9 * tau_from_surface.max()))])
        source_over_b = source / planck_plateau
        order = np.argsort(tau_from_surface)

        ax.loglog(
            tau_from_surface[order],
            source_over_b[order],
            color=color,
            lw=2.0,
            label=rf"SolRaT ($\epsilon={epsilon:.0e}$)",
        )
        # sqrt(epsilon) surface asymptote (the law's profile-independent content); the Doppler line
        # thermalizes to B over tau ~ 1/epsilon, not the monochromatic Eddington 1/sqrt(3 epsilon).
        ax.axhline(np.sqrt(epsilon), color=color, lw=1.2, ls=":")

        surface_value = float(source_over_b[np.argmin(tau_from_surface)])
        logging.info(
            "epsilon = %.0e: tau_total = %.2e, iterations = %d, residual = %.2e",
            epsilon,
            float(atmosphere.tau_grid[-1]),
            atmosphere.iterations_used,
            atmosphere.final_residual,
        )
        print(
            f"epsilon = {epsilon:.0e}: S(0)/B = {surface_value:.4f} vs sqrt(epsilon) = {np.sqrt(epsilon):.4f} "
            f"(ratio {surface_value / np.sqrt(epsilon):.3f})"
        )

    save_warm_state(__file__, atmosphere)

    ax.set_xlabel(r"optical depth from the surface  $\tau$")
    ax.set_ylabel(r"$S(\tau)\,/\,B$")
    ax.set_title(r"$\sqrt{\epsilon}$ thermalization law (dotted = $\sqrt{\epsilon}$ surface value)")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
    plt.show()
