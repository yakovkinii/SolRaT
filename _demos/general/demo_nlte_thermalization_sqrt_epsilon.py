import logging

import matplotlib.pyplot as plt
import numpy as np

from _demos.general.state.warm_start import load_warm_state
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

DOPPLER_LINE_CENTER_PROFILE = 1.0 / np.sqrt(np.pi)
AH65_FIG2_EPSILON_1EM2_TAU = np.array(
    [
        1.258925e-02,
        1.584893e-02,
        1.995262e-02,
        2.511886e-02,
        3.162278e-02,
        3.981072e-02,
        5.011872e-02,
        6.309573e-02,
        7.943282e-02,
        1.000000e-01,
        1.258925e-01,
        1.584893e-01,
        1.995262e-01,
        2.511886e-01,
        3.162278e-01,
        3.981072e-01,
        5.011872e-01,
        6.309573e-01,
        7.943282e-01,
        1.000000e00,
        1.258925e00,
        1.584893e00,
        1.995262e00,
        2.511886e00,
        3.162278e00,
        3.981072e00,
        5.011872e00,
        6.309573e00,
        7.943282e00,
        1.000000e01,
        1.258925e01,
        1.584893e01,
        1.995262e01,
        2.511886e01,
        3.162278e01,
        3.981072e01,
        5.011872e01,
        6.309573e01,
        7.943282e01,
        1.000000e02,
        1.258925e02,
        1.584893e02,
        1.995262e02,
        2.511886e02,
        3.162278e02,
    ]
)
AH65_FIG2_EPSILON_1EM2_SOURCE = np.array(
    [
        1.002998e-01,
        1.002998e-01,
        1.002998e-01,
        1.001383e-01,
        1.018826e-01,
        1.025180e-01,
        1.079692e-01,
        1.080936e-01,
        1.080936e-01,
        1.080936e-01,
        1.098753e-01,
        1.118922e-01,
        1.165199e-01,
        1.212830e-01,
        1.226027e-01,
        1.282331e-01,
        1.333829e-01,
        1.425936e-01,
        1.521248e-01,
        1.633804e-01,
        1.774598e-01,
        1.924864e-01,
        2.101842e-01,
        2.306747e-01,
        2.533961e-01,
        2.802851e-01,
        3.130401e-01,
        3.509942e-01,
        3.865449e-01,
        4.290422e-01,
        4.727157e-01,
        5.165353e-01,
        5.632484e-01,
        6.147521e-01,
        6.620640e-01,
        7.130172e-01,
        7.561361e-01,
        7.978110e-01,
        8.350260e-01,
        8.665627e-01,
        8.920722e-01,
        9.785883e-01,
        9.915166e-01,
        9.824263e-01,
        9.928874e-01,
    ]
)


def slab_height_for_tau_total(model, temperature_K, number_density_cm3, nu, target_tau_line_integrated_total):
    r"""
    Slab height [cm] giving the line-integrated optical thickness. SolRaT's
    ``tau_grid`` is the line-center optical depth, so for pure Doppler broadening
    :math:`\tau_c = \phi(0)\tau` with :math:`\phi(0)=1/\sqrt{\pi}`.
    """
    probe_height_cm = 1.0e9
    probe = NLTEStratifiedAtmosphere(
        model=model,
        stratification=StratifiedAtmosphere(
            model=model,
            height_cm=height_grid_refined_at_observer_surface(probe_height_cm, n_near_surface=10, n_interior=5),
            temperature_K=temperature_K,
            number_density_cm3=number_density_cm3,
            voigt_a=0.0,
        ),
        los_theta=0.0,
        n_mu_quadrature=2,
        n_phi_quadrature=3,
        max_iterations=1,
        tolerance=1.0,
    )
    probe.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    target_tau_line_center_total = DOPPLER_LINE_CENTER_PROFILE * target_tau_line_integrated_total
    return probe_height_cm * target_tau_line_center_total / float(probe.tau_grid[-1])


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


def benchmark_rms(tau, values, benchmark_tau, benchmark_values):
    mask = tau > 0.0
    order = np.argsort(tau[mask])
    model_at_benchmark = np.interp(np.log10(benchmark_tau), np.log10(tau[mask][order]), values[mask][order])
    return float(np.sqrt(np.mean((model_at_benchmark - benchmark_values) ** 2)))


def main(warm_start=True):
    r"""
    Line source function :math:`S(\tau)/B` for a pure-Doppler, finite-:math:`\epsilon`,
    isothermal, effectively semi-infinite two-level atom. The demo uses :math:`\epsilon=10^{-2}`
    and compares against the Avrett & Hummer (1965) Fig. 2 benchmark.

    :math:`B` is read as the thermalized interior value of :math:`S` itself (many thermalization
    depths below the observer surface), so the test is independent of the absolute intensity units.
    """
    setup_logging()

    temperature_K = 6000.0
    number_density_cm3 = 1.0e11
    epsilon = 1e-2
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

    collisions.set_deexcitation_rate_from_epsilon(transition=transition, epsilon=epsilon, temperature_K=temperature_K)
    initial_state = load_warm_state(__file__, warm_start)
    # Semi-infinite: total thickness >> the thermalization depth 1/epsilon, so the surface value
    # is the semi-infinite limit.
    slab_height_cm = slab_height_for_tau_total(
        model, temperature_K, number_density_cm3, nu, target_tau_line_integrated_total=100.0 / epsilon
    )
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(slab_height_cm, 4 * points_per_decade, 3 * points_per_decade),
        temperature_K=temperature_K,
        number_density_cm3=number_density_cm3,
        voigt_a=0.0,
    )
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=0.0,
        n_mu_quadrature=8,
        n_phi_quadrature=3,
        max_iterations=10000,
        tolerance=1e-10,
        ng_acceleration=True,
        ng_damping=0.5,
        ng_period=7,
        transfer_scheme="delo_linear",
        estimate_true_error=True,
    )
    # Thermalized (Planck) lower boundary: makes the slab semi-infinite, so only the observer
    # surface shows the sqrt-epsilon dip (a vacuum lower boundary would be a second free surface).
    atmosphere.forward(
        initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=temperature_K),
        initial_state=initial_state,
    )
    # save_warm_state(__file__, atmosphere)

    # tau_grid is line-center optical depth. The plotted tau is line-integrated optical depth,
    # measured from the observer surface inward.
    tau_line_center_from_surface = float(atmosphere.tau_grid[-1]) - atmosphere.tau_grid
    tau_from_surface = tau_line_center_from_surface / DOPPLER_LINE_CENTER_PROFILE
    source = line_center_source_function(atmosphere, nu, line_center)
    # Thermalized interior (deep, far below the observer surface) stands in for B, so the ratio is
    # unit-free; only the observer surface dips to sqrt(epsilon) B (the lower boundary is Planck).
    planck_plateau = float(source[np.argmin(np.abs(tau_from_surface - 0.9 * tau_from_surface.max()))])
    source_over_b = source / planck_plateau
    order = np.argsort(tau_from_surface)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(
        tau_from_surface[order],
        source_over_b[order],
        color="k",
        lw=1.4,
        label=rf"SolRaT ($\epsilon={epsilon:.0e}$)",
    )
    rms = benchmark_rms(tau_from_surface, source_over_b, AH65_FIG2_EPSILON_1EM2_TAU, AH65_FIG2_EPSILON_1EM2_SOURCE)
    ax.semilogx(
        AH65_FIG2_EPSILON_1EM2_TAU,
        AH65_FIG2_EPSILON_1EM2_SOURCE,
        linestyle="none",
        marker="x",
        markersize=5.5,
        markeredgewidth=1.3,
        color="k",
        label=rf"AH65 ($\epsilon={epsilon:.0e}$)",
    )
    # sqrt(epsilon) surface asymptote (the law's profile-independent content); the Doppler line
    # thermalizes to B over tau ~ 1/epsilon, not the monochromatic Eddington 1/sqrt(3 epsilon).
    ax.axhline(
        np.sqrt(epsilon),
        color="k",
        lw=1,
        ls="--",
        dash_capstyle="round",
        label=r"$S(0)/B=\sqrt{{\epsilon}}$",
    )

    surface_value = float(source_over_b[np.argmin(tau_from_surface)])
    logging.info(
        "epsilon = %.0e: line-integrated tau_total = %.2e, iterations = %d, residual = %.2e",
        epsilon,
        float(tau_from_surface[0]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )
    logging.info(
        "epsilon = %.0e: S(0)/B = %.4f vs sqrt(epsilon) = %.4f (ratio %.3f)",
        epsilon,
        surface_value,
        np.sqrt(epsilon),
        surface_value / np.sqrt(epsilon),
    )
    print(f"AH65 Fig. 2, epsilon={epsilon:.0e}: RMS S(tau)/B = {rms:.3e}")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$S(\tau)\,/\,B$")
    ax.set_ylim(0.05, 1.05)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.grid(color="0.88", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
    plt.show()
