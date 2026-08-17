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


def main():
    r"""
    Show how the emergent :math:`Q/I` profile of the TB1999 (:math:`\mu=0.1`) scattering line
    converges, first on a coarse spectral sampling and then continued -- via the warm-start
    :class:`NLTEState` API -- on a refined one, without restarting from the LTE guess.
    """
    setup_logging()

    temperature_K = 6000.0

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition=transition, epsilon=1e-2, temperature_K=temperature_K)

    params = model.AtmosphereParameters(model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K)
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_v = params.delta_v_thermal_cm_sm1
    nu_coarse = frequencies_around_line_sm1(nu0, delta_v, step_doppler=0.5)
    nu_fine = frequencies_around_line_sm1(nu0, delta_v, step_doppler=0.2)

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(1000e5, n_near_surface=200, n_interior=100),
        temperature_K=temperature_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(0.1)),  # mu
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=2000,
        tolerance=1e-9,
        ng_acceleration=True,
        ng_damping=0.5,
        ng_period=10,
    )

    atmosphere2 = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(0.1)),  # mu
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=20,
        n_phi_quadrature=3,
        max_iterations=2000,
        tolerance=1e-9,
        ng_acceleration=True,
        ng_damping=0.5,
        ng_period=10,
    )


    frames: List[dict] = []

    def recorder(reduced_nu: np.ndarray, phase_label: str, iteration_offset: int):
        def record(iteration: int, emergent: Stokes) -> None:
            residual = atmosphere.final_residual
            frames.append(
                {
                    "x": reduced_nu,
                    "y": 100.0 * emergent.Q / emergent.I,
                    "iteration": iteration_offset + iteration,
                    "residual": residual,
                    "title": rf"{phase_label}, $\Lambda$-iteration {iteration_offset + iteration} "
                    rf"($\max|\Delta\rho|$={residual:.1e})",
                }
            )

        return record

    initial_state = NLTEState.load("converged_state.npz")
    atmosphere.forward(
        initial_stokes=Stokes.from_zeros(nu_sm1=nu_coarse),
        initial_state=initial_state,
        on_iteration=recorder(reduced_frequency(nu_coarse, nu0, delta_v), r"coarse (0.5 $\Delta\nu_D$)", 1),
    )
    n_coarse = len(frames)
    state_coarse = atmosphere.get_state()

    atmosphere2.forward(
        initial_stokes=Stokes.from_zeros(nu_sm1=nu_fine),
        initial_state=state_coarse,
        on_iteration=recorder(reduced_frequency(nu_fine, nu0, delta_v), r"fine (0.1 $\Delta\nu_D$)", n_coarse + 1),
    )
    converged_state = atmosphere2.get_state()
    converged_state.save("converged_state.npz")
    converged_y = frames[-1]["y"]
    last_coarse_iteration = frames[n_coarse - 1]["iteration"]

    fig, ax = plt.subplots(figsize=(7, 5))
    colormap = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=1, vmax=frames[-1]["iteration"])
    for frame in frames[:-1]:
        style = "--" if frame["iteration"] <= last_coarse_iteration else "-"
        ax.plot(frame["x"], frame["y"], lw=1.0, ls=style, color=colormap(normalizer(frame["iteration"])))
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

    line_center = int(np.argmin(np.abs(reduced_frequency(nu_fine, nu0, delta_v))))
    print(
        f"TB1999 mu=0.1 convergence evolution: coarse phase {n_coarse} iterations to "
        f"{frames[n_coarse - 1]['residual']:.1e}, fine phase {len(frames) - n_coarse} iterations to "
        f"{frames[-1]['residual']:.1e}; line-center 100 Q/I = {converged_y[line_center]:.3f}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
