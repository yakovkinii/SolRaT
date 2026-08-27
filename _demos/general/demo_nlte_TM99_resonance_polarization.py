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
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    frequencies_around_line_sm1,
    height_grid_refined_at_observer_surface,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


def upper_level_alignment(atmosphere: NLTEStratifiedAtmosphere, upper_level_id: str) -> np.ndarray:
    r"""
    Fractional atomic alignment :math:`\rho^2_0 / \rho^0_0` of the upper level over the depth grid
    (TM99 Fig. 8).
    """
    sigma = []
    for rho in atmosphere.rho_grid:
        rho00 = np.real(rho(K=0, Q=0, level_id=upper_level_id))
        rho20 = np.real(rho(K=2, Q=0, level_id=upper_level_id))
        sigma.append(rho20 / rho00)
    return np.array(sigma)


def benchmark_rms(tau, alignment, benchmark_tau, benchmark_alignment):
    mask = tau > 0.0
    order = np.argsort(tau[mask])
    model_at_benchmark = np.interp(
        np.log10(benchmark_tau),
        np.log10(tau[mask][order]),
        alignment[mask][order],
    )
    return float(np.sqrt(np.mean((model_at_benchmark - benchmark_alignment) ** 2)))


def calculate_alignment_for_delta2(delta2: float, warm_start: bool):
    temperature_K = 6000.0
    epsilon = 1.0e-4
    mu_observer = 0.0
    n_mu_tm99 = 5

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    upper_level_id = transition.level_upper.level_id
    collisions.set_deexcitation_rate_from_epsilon(transition, epsilon, temperature_K)
    collisions.set_depolarizing_rate(upper_level_id, K=2, rate_sm1=delta2 * transition.einstein_a_ul)

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(), params.delta_v_thermal_cm_sm1, step_doppler=0.5
    )
    line_center_index = int(np.argmin(np.abs(nu - transition.get_mean_transition_frequency_sm1())))

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(
            1000e6, n_near_surface=100, n_interior=30, min_surface_fraction=1e-9
        ),
        temperature_K=temperature_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    state_suffix = f"_delta2_{delta2:g}".replace(".", "p")
    initial_state = load_warm_state(__file__, warm_start, suffix=state_suffix)
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(mu_observer)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=2 * n_mu_tm99,
        n_phi_quadrature=3,
        max_iterations=100,
        tolerance=1e-8,
        ng_acceleration=True,
        ng_period=7,
        ng_damping=1,
        transfer_scheme="delo_linear",
        estimate_true_error=True,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu), initial_state=initial_state)
    # save_warm_state(__file__, atmosphere, suffix=state_suffix)

    vertical_tau_line_center = atmosphere.tau_grid
    tau_from_surface = (vertical_tau_line_center[-1] - vertical_tau_line_center) * np.sqrt(np.pi)
    alignment = upper_level_alignment(atmosphere, upper_level_id)
    surface_alignment = alignment[-1]
    emergent_qi_percent = 100.0 * emergent.Q[line_center_index] / emergent.I[line_center_index]

    logging.info("TM99 benchmark: epsilon = %.0e, delta2 = %.1f, no continuum, no field", epsilon, delta2)
    logging.info(
        "line-integrated optical thickness = %.1f, iterations = %d, residual = %.2e",
        float(tau_from_surface[0]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )
    logging.info(
        f"TM99 (epsilon={epsilon:.0e}, delta2={delta2:.1f}): surface rho^2_0/rho^0_0 = "
        f"{surface_alignment:.5f}; tangential Q/I = {emergent_qi_percent:.3f}%; "
        f"iterations = {atmosphere.iterations_used}"
    )
    return tau_from_surface, alignment


def main(warm_start=True):
    r"""
    Reproduce the Trujillo Bueno & Manso Sainz (1999), ApJ 516, 436, resonance-line-polarization
    benchmark: a :math:`J=0 \to 1` two-level atom in an isothermal, self-emitting, plane-parallel
    slab, with the photon destruction probability set through the parametrized collisional
    de-excitation rate.

    Plots the upper-level alignment :math:`\rho^2_0/\rho^0_0` versus optical depth from the surface
    against digitized TM99 Fig. 8 curves for two depolarizing rates.
    """
    setup_logging()

    fig_alignment, ax_alignment = plt.subplots(figsize=(7, 5))
    tm99_depth = np.array(
        [
            1.524109e-04,
            2.023033e-04,
            2.685281e-04,
            3.564319e-04,
            4.731114e-04,
            6.279864e-04,
            8.335603e-04,
            1.106430e-03,
            1.468624e-03,
            1.949385e-03,
            2.587524e-03,
            3.434560e-03,
            4.558878e-03,
            6.051246e-03,
            8.032147e-03,
            1.066150e-02,
            1.415159e-02,
            1.878417e-02,
            2.493325e-02,
            3.309526e-02,
            4.392913e-02,
            5.830951e-02,
            7.739737e-02,
            1.027337e-01,
            1.363640e-01,
            1.810034e-01,
            2.402556e-01,
            3.189043e-01,
            4.232989e-01,
            5.618676e-01,
            7.457973e-01,
            9.899371e-01,
            1.313997e00,
            1.744140e00,
            2.315091e00,
            3.072946e00,
            4.078888e00,
            5.414128e00,
            7.186466e00,
            9.538986e00,
            1.266161e01,
            1.680644e01,
            2.230810e01,
            2.961076e01,
            3.930396e01,
            5.217028e01,
            6.924843e01,
            9.191720e01,
            1.220067e02,
            1.619461e02,
            2.149598e02,
            2.853278e02,
            3.787310e02,
            5.027102e02,
            6.672745e02,
            8.857096e02,
            1.175650e03,
            1.560504e03,
            2.071342e03,
            2.749405e03,
            3.649434e03,
            4.844091e03,
            6.429825e03,
            8.534654e03,
            1.132851e04,
            1.503694e04,
            1.995935e04,
            2.649313e04,
            3.516576e04,
            4.667742e04,
            6.195747e04,
            8.223951e04,
            1.091610e05,
            1.448953e05,
            1.923273e05,
            2.552865e05,
            3.388556e05,
            4.497814e05,
            5.970192e05,
            7.924559e05,
            1.051870e06,
            1.396204e06,
            1.853257e06,
            2.459928e06,
            3.265196e06,
        ]
    )
    tm99_alignment_1 = np.array(
        [
            0.036886,
            0.036851,
            0.036825,
            0.036825,
            0.036825,
            0.036801,
            0.036729,
            0.036668,
            0.036668,
            0.036622,
            0.036481,
            0.036390,
            0.036225,
            0.036021,
            0.035750,
            0.035424,
            0.034962,
            0.034395,
            0.033661,
            0.032728,
            0.031569,
            0.030166,
            0.028450,
            0.026466,
            0.024245,
            0.021755,
            0.019049,
            0.016279,
            0.013390,
            0.010620,
            0.007956,
            0.005524,
            0.003580,
            0.002010,
            0.000904,
            0.000187,
            -0.000104,
            -0.000180,
            -0.000213,
            -0.000124,
            -0.000041,
            0.000002,
            0.000035,
            -0.000041,
            -0.000083,
            -0.000083,
            -0.000083,
            -0.000082,
            -0.000052,
            -0.000023,
            0.000011,
            0.000064,
            0.000064,
            0.000073,
            0.000077,
            0.000077,
            0.000079,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000116,
        ]
    )

    tm99_alignment_0dot1 = np.array(
        [  # for delta(2) = 0.1
            0.073423,
            0.073362,
            0.073362,
            0.073362,
            0.073332,
            0.073332,
            0.073271,
            0.073175,
            0.073088,
            0.072998,
            0.072837,
            0.072618,
            0.072366,
            0.072046,
            0.071582,
            0.070981,
            0.070207,
            0.069225,
            0.067933,
            0.066320,
            0.064278,
            0.061864,
            0.058797,
            0.055291,
            0.051281,
            0.046601,
            0.041622,
            0.036374,
            0.030692,
            0.025000,
            0.019329,
            0.014102,
            0.009565,
            0.005672,
            0.002767,
            0.000954,
            -0.000289,
            -0.001112,
            -0.001304,
            -0.001223,
            -0.001064,
            -0.000914,
            -0.000673,
            -0.000532,
            -0.000083,
            -0.000083,
            -0.000083,
            -0.000082,
            -0.000052,
            -0.000023,
            0.000011,
            0.000064,
            0.000064,
            0.000073,
            0.000077,
            0.000077,
            0.000079,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
            0.000097,
        ]
    )

    benchmark_by_delta2 = {
        1.0: tm99_alignment_1,
        0.1: tm99_alignment_0dot1,
    }
    colors_by_delta2 = {
        1.0: "k",
        0.1: "#d62728",
    }
    solrat_by_delta2 = {
        delta2: calculate_alignment_for_delta2(delta2=delta2, warm_start=warm_start) for delta2 in benchmark_by_delta2
    }

    for delta2, benchmark_alignment in benchmark_by_delta2.items():
        color = colors_by_delta2[delta2]
        ax_alignment.plot(
            tm99_depth[: len(benchmark_alignment)],
            benchmark_alignment,
            linestyle="none",
            marker="x",
            markersize=5.5,
            markeredgewidth=1.3,
            color=color,
            label=rf"TM99 ($\delta^{{(2)}}={delta2:g}$)",
        )
        tau_from_surface, alignment = solrat_by_delta2[delta2]
        rms = benchmark_rms(
            tau_from_surface,
            alignment,
            tm99_depth[: len(benchmark_alignment)],
            benchmark_alignment,
        )
        ax_alignment.plot(
            tau_from_surface[1:-1],
            alignment[1:-1],
            "-",
            color=color,
            lw=1.8,
            label=rf"SolRaT ($\delta^{{(2)}}={delta2:g}$)",
        )
        print(f"TM99 Fig. 8, delta2={delta2:g}: RMS rho^2_0/rho^0_0 = {rms:.3e}")

    ax_alignment.set_xscale("log")
    ax_alignment.set_xlim(1e-3, 1e5)
    ax_alignment.set_xlabel(r"$\tau$")
    ax_alignment.set_ylabel(r"$\rho^2_0 / \rho^0_0$")
    # ax_alignment.set_ylim(-0.02, 0.10)
    ax_alignment.grid(color="0.88", linewidth=0.5, alpha=0.7)
    ax_alignment.legend(loc="best")
    fig_alignment.tight_layout()

    return fig_alignment


if __name__ == "__main__":
    main()
    plt.show()
