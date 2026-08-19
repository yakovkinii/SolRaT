import logging
import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from matplotlib.colors import Normalize

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.nlte_state import NLTEState
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    frequencies_around_line_sm1,
    height_grid_refined_at_observer_surface,
    reduced_frequency,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


def save_convergence_animation(
    frames: List[dict], converged_y: np.ndarray, output_path: pathlib.Path, fps: int
) -> None:
    r"""
    Best-effort: render the recorded ``100 Q/I`` frames as a GIF at ``fps`` frames per second. Skipped
    with a log message if no Matplotlib animation writer (e.g. Pillow) is available.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.axhline(0.0, color="0.7", lw=0.6)
    axis.plot(frames[-1]["x"], converged_y, lw=2.0, ls="--", color="0.6", label="converged")
    (line,) = axis.plot([], [], lw=2.0, color="k")
    x_all = np.concatenate([frame["x"] for frame in frames])
    y_all = np.concatenate([frame["y"] for frame in frames] + [converged_y])
    axis.set_xlim(float(x_all.min()), float(x_all.max()))
    axis.set_ylim(float(y_all.min()) - 0.2, float(y_all.max()) + 0.2)
    axis.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    axis.set_ylabel(r"$100\,Q/I$")
    axis.legend(loc="upper right")

    hold = [frames[-1]] * 10

    def update(frame_index):
        frame = (frames + hold)[frame_index]
        line.set_data(frame["x"], frame["y"])
        axis.set_title(frame["title"])
        return (line,)

    anim = animation.FuncAnimation(figure, update, frames=len(frames) + len(hold), blit=False)
    try:
        anim.save(str(output_path), writer=animation.PillowWriter(fps=fps))
        logging.info("Saved convergence animation to %s", output_path)
    except Exception as exc:  # pragma: no cover - animation-writer availability is environment-dependent
        logging.warning("Could not save convergence animation (%s); the static figure is still produced.", exc)
    plt.close(figure)


def slab_height_for_tau_total(model, temperature_K, number_density_cm3, los_theta, nu, target_tau_total):
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
        los_theta=los_theta,
        n_mu_quadrature=2,
        n_phi_quadrature=1,
        max_iterations=1,
        tolerance=1.0,
    )
    probe.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    return probe_height_cm * target_tau_total / float(probe.tau_grid[-1])


def main():
    r"""
    Emergent :math:`Q/I` profile of the TB1999 (:math:`\mu=0.1`) scattering line, converged from the
    saved :class:`NLTEState` of the previous run (warm-started, not from the LTE guess) and overlaid on
    the digitized TB1999 Fig. 10.
    """
    setup_logging()

    temperature_K = 6000.0
    epsilon = 1.0e-2
    mu = 0.1
    number_density_cm3 = 1.0e11
    target_tau_total = 1.0e4  # >> thermalization depth 1/epsilon = 100, so the surface value is the semi-infinite limit
    points_per_decade = 80  # TB1999 converge their grids at ~23-46 points per decade of optical depth (their Tables 1-3)
    n_near_surface = 4 * points_per_decade  # the 1e-7..1e-3 surface segment spans 4 decades
    n_interior = 3 * points_per_decade  # the 1e-3..1 interior segment spans 3 decades

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition=transition, epsilon=epsilon, temperature_K=temperature_K)

    params = model.AtmosphereParameters(model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K)
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_v = params.delta_v_thermal_cm_sm1
    nu = frequencies_around_line_sm1(nu0, delta_v, half_width_doppler=5.0, step_doppler=0.1)

    slab_height_cm = slab_height_for_tau_total(
        model, temperature_K, number_density_cm3, float(np.arccos(mu)), nu, target_tau_total
    )
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(slab_height_cm, n_near_surface, n_interior),
        temperature_K=temperature_K,
        number_density_cm3=number_density_cm3,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(mu)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=100,
        n_phi_quadrature=3,
        max_iterations=2000,
        tolerance=1e-9,
        ng_acceleration=True,
        ng_damping=0.5,
        ng_period=10,
        transfer_scheme="delo_linear",
        estimate_true_error=True,
    )

    frames: List[dict] = []
    reduced_nu = reduced_frequency(nu, nu0, delta_v)

    def record(iteration: int, emergent: Stokes) -> None:
        residual = atmosphere.final_residual
        frames.append(
            {
                "x": reduced_nu,
                "y": 100.0 * emergent.Q / emergent.I,
                "iteration": iteration + 1,
                "residual": residual,
                "title": rf"$\Lambda$-iteration {iteration + 1} ($\max|\Delta\rho|$={residual:.1e})",
            }
        )

    initial_state = NLTEState.load("converged_state.npz")
    atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu), initial_state=initial_state, on_iteration=record)
    atmosphere.get_state().save("converged_state.npz")
    converged_y = frames[-1]["y"]

    fig, ax = plt.subplots(figsize=(7, 5))
    colormap = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=1, vmax=frames[-1]["iteration"])
    for frame in frames[:-1]:
        ax.plot(frame["x"], frame["y"], lw=1.0, color=colormap(normalizer(frame["iteration"])))
    ax.plot(frames[-1]["x"], converged_y, lw=2.6, color="k", label=f"converged (iteration {frames[-1]['iteration']})")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    ax.set_ylabel(r"$100\,Q/I$")

    # TB1999 Fig. 10 (delta2 = 0, mu = 0.1) digitized, blue wing mirrored onto the red.
    tb_reduced_frequency = np.array([
        -5.00365, -4.75456, -4.39872, -4.05109, -3.71989, -3.37226, -3.07664, -2.84672, -2.63869,
        -2.43066, -2.23905, -2.01734, -1.81752, -1.61496, -1.42336, -1.24818, -1.06752, -0.9115,
        -0.82391, -0.62956, -0.48996, -0.4188, -0.30109, -0.23266, -0.02737,
    ])
    tb_qi_percent = np.array([
        0.0, -0.00226, -0.00792, -0.00792, -0.00226, -0.01075, 0.01188, 0.0543, 0.13348, 0.28337,
        0.50113, 0.77828, 0.86312, 0.6086, 0.04581, -0.65554, -1.39367, -1.93665, -2.29016, -2.7681,
        -3.01697, -3.13292, -3.25735, -3.33371, -3.40158,
    ])
    tb_reduced_frequency_full = np.concatenate([tb_reduced_frequency, -tb_reduced_frequency[::-1]])
    tb_qi_percent_full = np.concatenate([tb_qi_percent, tb_qi_percent[::-1]])

    ax.plot(
        tb_reduced_frequency_full,
        tb_qi_percent_full,
        linestyle="none",
        marker="x",
        color="k",
        label="TB1999 Fig. 10 (digitized)",
    )

    scalar_mappable = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=ax, label=r"$\Lambda$-iteration")
    ax.legend()
    fig.tight_layout()

    save_convergence_animation(
        frames, converged_y, pathlib.Path(__file__).with_name("tb1999_convergence_evolution.gif"), fps=10
    )

    line_center = int(np.argmin(np.abs(reduced_nu)))
    tb_on_solrat = np.interp(reduced_nu, tb_reduced_frequency_full, tb_qi_percent_full)
    rms = float(np.sqrt(np.mean((converged_y - tb_on_solrat) ** 2)))
    print(
        f"TB1999 mu=0.1 match: tau_total = {float(atmosphere.tau_grid[-1]):.3e} (target {target_tau_total:.0e}), "
        f"n_mu = {atmosphere.n_mu_quadrature}, {points_per_decade} pts/decade; "
        f"{atmosphere.iterations_used} iters (residual {frames[-1]['residual']:.1e}); "
        f"line-center 100 Q/I = {converged_y[line_center]:.4f} vs TB1999 {tb_on_solrat[line_center]:.4f}; "
        f"RMS(profile) = {rms:.4f}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
