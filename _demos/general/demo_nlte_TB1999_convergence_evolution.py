import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm
from matplotlib.colors import Normalize
from numpy import exp

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, kB_erg_Km1
from solrat.atom_model.shared.utility.log_setup import setup_logging

TEMPERATURE_K = 6000.0  # isothermal slab, same benchmark as demo_nlte_TB1999_resonance_polarization_mu01
EPSILON = 1.0e-2  # TB1999 photon destruction probability
MU_OBSERVER = 0.1  # inclined line of sight (mu = 0.1), the emergent Q/I profile of TB1999 Fig. 10
FRAME_ITERATIONS = tuple(range(2, 19, 2))  # Lambda-iterations captured as evolution frames: 2, 4, ..., 18
CONVERGED_ITERATION = 20  # this many plain iterations is taken as the converged reference profile


def c_ul_for_epsilon(epsilon: float, transition, temperature_K: float) -> float:
    r"""
    Collisional de-excitation rate :math:`C_{ul}` [1/s] that yields a two-level-atom photon
    destruction probability ``epsilon`` (LL04 Sec. 7.13; Mihalas 1978; TB1999 Sec. 2):
    :math:`C_{ul} = \frac{\epsilon}{1-\epsilon}\, A_{ul} / (1 - e^{-h\nu_0/kT})`.

    :param epsilon: photon destruction probability in (0, 1) [dimensionless].
    :param transition: the radiative transition (carries the Einstein A and level energies).
    :param temperature_K: local temperature [K].
    :return: collisional de-excitation rate [1/s].
    """
    assert 0.0 < epsilon < 1.0, "epsilon must be in (0, 1)."
    delta_e_erg = (transition.level_upper.energy_cmm1 - transition.level_lower.energy_cmm1) * h_erg_s * c_cm_sm1
    stimulated_correction = 1.0 - exp(-delta_e_erg / (kB_erg_Km1 * temperature_K))
    return epsilon / (1.0 - epsilon) * transition.einstein_a_ul / stimulated_correction


def surface_refined_depth_grid(z_max_cm: float, n_surface: int, n_deep: int) -> np.ndarray:
    r"""
    Depth grid concentrated near the observer surface (where the inclined-ray line core forms), with
    a sparse thermalized interior. ``z[0]`` is the lower boundary, ``z[-1]`` the observer surface.

    :param z_max_cm: slab thickness [cm].
    :param n_surface: number of logarithmically packed surface points.
    :param n_deep: number of sparse interior points.
    :return: sorted height grid [cm].
    """
    surface = np.logspace(np.log10(1e-7), np.log10(1e-3), n_surface, endpoint=False)
    deep = np.logspace(np.log10(1e-3), 0.0, n_deep)
    depth_below_surface = z_max_cm * np.concatenate([surface, deep])
    return np.sort(z_max_cm - depth_below_surface)


def build_frequency_grid(transition, delta_v_thermal_cm_sm1: float) -> np.ndarray:
    r"""
    Frequency grid at ~10 points per Doppler width over +-4 Doppler widths.

    :param transition: the radiative transition.
    :param delta_v_thermal_cm_sm1: thermal+turbulent Doppler velocity [cm/s].
    :return: frequency grid [1/s].
    """
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
    step = 0.1 * delta_nu_D
    return np.arange(nu0 - 4.0 * delta_nu_D, nu0 + 4.0 * delta_nu_D + 0.5 * step, step)


def emergent_qi_after_k_iterations(atmosphere: NLTEStratifiedAtmosphere, nu: np.ndarray, k: int) -> np.ndarray:
    r"""
    Emergent line-of-sight ``100 Q/I`` profile after exactly ``k`` plain Lambda-iterations.

    ``forward`` always restarts from the same isotropic-Planck guess and (with the tolerance set
    effectively to zero) runs the full ``max_iterations``, so setting ``max_iterations = k`` and
    re-running gives the deterministic k-th iterate without touching the solver internals.

    :param atmosphere: the NLTE atmosphere (Ng disabled, tolerance ~ 0 so it never stops early).
    :param nu: frequency grid [1/s].
    :param k: number of Lambda-iterations.
    :return: ``100 Q/I`` over the frequency grid.
    """
    atmosphere.max_iterations = k
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    return 100.0 * emergent.Q / emergent.I


def save_convergence_animation(
    reduced_frequency: np.ndarray,
    frame_profiles: list,
    frame_titles: list,
    qi_converged: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    r"""
    Best-effort: render the ``100 Q/I`` profiles as a GIF at ``output_path``. Skipped with a log
    message if no Matplotlib animation writer (e.g. Pillow) is available.

    :param reduced_frequency: reduced-frequency axis ``(nu - nu0)/Delta nu_D``.
    :param frame_profiles: list of ``100 Q/I`` arrays, one per animation frame (in display order).
    :param frame_titles: per-frame titles, same length as ``frame_profiles``.
    :param qi_converged: converged ``100 Q/I`` array (drawn as a fixed dashed reference).
    :param output_path: GIF destination.
    """
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.axhline(0.0, color="0.7", lw=0.6)
    axis.plot(reduced_frequency, qi_converged, lw=2.0, ls="--", color="0.6", label="converged")
    (line,) = axis.plot([], [], lw=2.0, color="k")
    axis.set_xlim(float(reduced_frequency.min()), float(reduced_frequency.max()))
    y_all = np.concatenate(list(frame_profiles) + [qi_converged])
    axis.set_ylim(float(y_all.min()) - 0.2, float(y_all.max()) + 0.2)
    axis.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    axis.set_ylabel(r"$100\,Q/I$")
    axis.legend(loc="upper right")

    def update(frame_index):
        line.set_data(reduced_frequency, frame_profiles[frame_index])
        # The per-frame title is the dynamic iteration counter (the point of the animation), not a
        # static figure title.
        axis.set_title(frame_titles[frame_index])
        return (line,)

    anim = animation.FuncAnimation(figure, update, frames=len(frame_profiles), blit=False)
    try:
        anim.save(str(output_path), writer=animation.PillowWriter(fps=2))
        logging.info("Saved convergence animation to %s", output_path)
    except Exception as exc:  # pragma: no cover - animation-writer availability is environment-dependent
        logging.warning("Could not save convergence animation (%s); the static figure is still produced.", exc)
    plt.close(figure)


def main():
    r"""
    Show how the emergent ``Q/I`` profile of the TB1999 (mu = 0.1) scattering line evolves as the
    self-consistent NLTE loop converges: from the isotropic-Planck guess (essentially unpolarized) the
    upper-level alignment builds up over the Lambda-iterations and the line-center ``Q/I`` grows toward
    its converged value. A companion of ``demo_nlte_TB1999_resonance_polarization_mu01`` (which shows
    only the final profile).

    The static figure overlays the ``100 Q/I`` profile at each iteration (light to dark) with the
    converged profile in bold black; a GIF of the same sequence is written next to this file
    (best-effort). Ng acceleration is disabled so the sequence is a clean monotone Lambda progression.

    :return: the matplotlib Figure with the overlaid per-iteration profiles (not shown; the caller
        decides whether to display it interactively or save it).
    """
    setup_logging()

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate(transition.transition_id, c_ul_for_epsilon(EPSILON, transition, TEMPERATURE_K))

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
    )
    nu = build_frequency_grid(transition, params.delta_v_thermal_cm_sm1)
    nu0 = transition.get_mean_transition_frequency_sm1()
    reduced_frequency = (nu - nu0) / (nu0 * params.delta_v_thermal_cm_sm1 / c_cm_sm1)

    # Same grid/quadrature as demo_nlte_TB1999_resonance_polarization_mu01: a coarser rule leaves the
    # inclined-ray reconstruction under-resolved (a spurious anisotropy J^2_0 > J^0_0), which blows the
    # alignment up toward Q = -I. The k-th iterate is obtained by re-running from scratch, so the total
    # cost grows quadratically in the highest captured iteration -- deliberately long, but robust.
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=surface_refined_depth_grid(1000e5, n_surface=80, n_deep=30),
        temperature_K=TEMPERATURE_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )

    frame_atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(MU_OBSERVER)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=1,
        tolerance=1e-30,  # effectively zero: never stop early, so max_iterations = k runs exactly k steps
        ng_acceleration=False,
    )
    qi_frames = [emergent_qi_after_k_iterations(frame_atmosphere, nu, k) for k in FRAME_ITERATIONS]
    qi_converged = emergent_qi_after_k_iterations(frame_atmosphere, nu, CONVERGED_ITERATION)

    fig, ax = plt.subplots(figsize=(7, 5))
    colormap = plt.get_cmap("viridis")
    normalizer = Normalize(vmin=FRAME_ITERATIONS[0], vmax=FRAME_ITERATIONS[-1])
    for k, qi in zip(FRAME_ITERATIONS, qi_frames):
        ax.plot(reduced_frequency, qi, lw=1.0, color=colormap(normalizer(k)))
    ax.plot(reduced_frequency, qi_converged, lw=2.6, color="k", label=f"converged (iteration {CONVERGED_ITERATION})")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    ax.set_ylabel(r"$100\,Q/I$")
    scalar_mappable = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=ax, label=r"$\Lambda$-iteration")
    ax.legend()
    fig.tight_layout()

    # Animation frame sequence: the step-2 iterates, then the converged profile, with the last frame
    # duplicated so the GIF holds on the converged state for a moment before looping.
    animation_profiles = list(qi_frames) + [qi_converged, qi_converged]
    animation_titles = [rf"$\Lambda$-iteration {k}" for k in FRAME_ITERATIONS] + [
        f"converged (iteration {CONVERGED_ITERATION})",
        f"converged (iteration {CONVERGED_ITERATION})",
    ]
    save_convergence_animation(
        reduced_frequency,
        animation_profiles,
        animation_titles,
        qi_converged,
        pathlib.Path(__file__).with_name("tb1999_convergence_evolution.gif"),
    )

    line_center = int(np.argmin(np.abs(reduced_frequency)))
    print(
        f"TB1999 mu=0.1 convergence evolution (iterations {FRAME_ITERATIONS[0]}..{FRAME_ITERATIONS[-1]} "
        f"step 2): line-center 100 Q/I grows from {qi_frames[0][line_center]:.3f} to "
        f"{qi_converged[line_center]:.3f} (converged, iteration {CONVERGED_ITERATION})"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
