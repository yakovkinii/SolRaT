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


def log_depth_grid(z_max_cm: float, n_depth: int, min_fraction: float = 1e-9) -> np.ndarray:
    r"""
    Height grid with the depth below the observer surface logarithmically spaced.
    """
    depth_below_surface = np.logspace(np.log10(z_max_cm * min_fraction), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth_below_surface)


def build_atmosphere(model, stratification, mu_observer, ng_acceleration, max_iterations):
    r"""
    A TB1999 self-consistent atmosphere with or without Ng acceleration (all other settings shared).
    """
    return NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=float(np.arccos(mu_observer)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=max_iterations,
        tolerance=1e-8,
        ng_acceleration=ng_acceleration,
        ng_damping=0.7,
    )


def main():
    r"""
    Convergence history of the TB1999 :math:`J=0 \to 1` resonance-line solve: the per-iteration
    residual :math:`\max|\Delta\rho|` of a plain :math:`\Lambda`-iteration against the Ng-accelerated
    one, both cold-started from the LTE guess. This is a solver diagnostic, not a physics benchmark;
    it is deliberately not used as a manuscript figure, and it must run cold, so it does not warm-start.
    """
    setup_logging()

    temperature_K = 6000.0
    epsilon = 1.0e-2
    mu_observer = 0.0

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition, epsilon, temperature_K)

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(), params.delta_v_thermal_cm_sm1, step_doppler=0.5
    )

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

    atmosphere_ng = build_atmosphere(model, stratification, mu_observer, ng_acceleration=True, max_iterations=1000)
    atmosphere_plain = build_atmosphere(model, stratification, mu_observer, ng_acceleration=False, max_iterations=2000)
    atmosphere_ng.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    atmosphere_plain.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(
        np.arange(1, len(atmosphere_plain.residual_history) + 1),
        atmosphere_plain.residual_history,
        marker=".",
        label=r"plain $\Lambda$-iteration",
    )
    ax.semilogy(
        np.arange(1, len(atmosphere_ng.residual_history) + 1),
        atmosphere_ng.residual_history,
        marker=".",
        label="Ng accelerated",
    )
    ax.axhline(atmosphere_ng.tolerance, color="k", linestyle="--", label=f"tolerance = {atmosphere_ng.tolerance:.0e}")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"convergence residual  $\max|\Delta\rho|$")
    ax.legend()
    fig.tight_layout()

    logging.info("TB1999 convergence history: epsilon = %.0e", epsilon)
    print(
        f"TB1999 convergence (epsilon={epsilon:.0e}): Ng iterations = {atmosphere_ng.iterations_used}, "
        f"plain iterations = {atmosphere_plain.iterations_used}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
