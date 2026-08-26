"""
TODO: this self-consistent NLTE depolarization scan is not converged on the short demo settings.
Launch a long run before using it as a quantitative validation.
"""

import numpy as np
from matplotlib import pyplot as plt

from _demos.general.state.warm_start import load_warm_state, save_warm_state
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

TEMPERATURE_K = 6000.0
NUMBER_DENSITY_CM3 = 1.0e11
SLAB_THICKNESS_CM = 1000e6


def surface_alignment(depolarizing_rate_over_a_ul: float, warm_start: bool) -> float:
    r"""
    Surface upper-level fractional alignment :math:`\rho^2_0/\rho^0_0` of a collisionless-scattering
    two-level atom with an elastic depolarizing rate :math:`D^{(2)}` on the upper level (LL04
    Sec. 7.13 / 10.6).

    :param depolarizing_rate_over_a_ul: ratio :math:`D^{(2)}/A_{ul}` [dimensionless].
    :return: surface :math:`\rho^2_0/\rho^0_0`.
    """
    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    upper_level_id = transition.level_upper.level_id
    collisions.set_depolarizing_rate(
        upper_level_id, K=2, rate_sm1=depolarizing_rate_over_a_ul * transition.einstein_a_ul
    )

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(), params.delta_v_thermal_cm_sm1, step_doppler=0.5
    )
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(SLAB_THICKNESS_CM, n_near_surface=50, n_interior=15),
        temperature_K=TEMPERATURE_K,
        number_density_cm3=NUMBER_DENSITY_CM3,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    state_suffix = f"_delta2_{depolarizing_rate_over_a_ul:g}".replace(".", "p")
    initial_state = load_warm_state(__file__, warm_start, suffix=state_suffix)
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=np.pi / 2,
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=6,
        n_phi_quadrature=3,
        max_iterations=500,
        tolerance=1e-8,
        ng_acceleration=True,
        ng_period=7,
        ng_damping=0.7,
        transfer_scheme="delo_linear",
        estimate_true_error=True,
    )
    atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu), initial_state=initial_state)
    save_warm_state(__file__, atmosphere, suffix=state_suffix)
    rho_surface = atmosphere.rho_grid[-1]
    return float(
        np.real(rho_surface(K=2, Q=0, level_id=upper_level_id))
        / np.real(rho_surface(K=0, Q=0, level_id=upper_level_id))
    )


def main(warm_start=True):
    r"""
    Collisional depolarization of resonance-scattering polarization (LL04 Sec. 10.6).

    Scans :math:`D^{(2)}/A_{ul}` for a :math:`J=0 \to 1` scattering slab and overlays the SolRaT
    surface alignment ratio on the analytic factor :math:`1/(1 + D^{(2)}/A_{ul})`.

    :return: matplotlib Figure.
    """
    setup_logging()

    depolarizing_ratios = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    alignments = np.array([surface_alignment(ratio, warm_start=warm_start) for ratio in depolarizing_ratios])
    normalized_alignment = alignments / alignments[0]
    analytic_factor = 1.0 / (1.0 + depolarizing_ratios)

    rms = float(np.sqrt(np.mean((normalized_alignment - analytic_factor) ** 2)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(depolarizing_ratios, normalized_alignment, lw=1.8, color="k", marker="x", label="SolRaT")
    ax.plot(
        depolarizing_ratios,
        analytic_factor,
        lw=2.4,
        ls=(0, (1, 1)),
        color="#d62728",
        label=r"$1/(1 + D^{(2)}/A_{ul})$",
    )
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"$D^{(2)}/A_{ul}$")
    ax.set_ylabel(r"$[\rho^2_0(D^{(2)})/\rho^0_0]/[\rho^2_0(0)/\rho^0_0]$")
    ax.grid(color="0.88", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    print(f"Collisional depolarization: RMS SolRaT - 1/(1+D2/A_ul) = {rms:.2e}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
