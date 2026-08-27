import logging

import matplotlib.pyplot as plt
import numpy as np

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
    reduced_frequency,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


def benchmark_rms(reduced_nu, qi_profile_percent, benchmark_reduced_nu, benchmark_qi_percent):
    order = np.argsort(reduced_nu)
    model_at_benchmark = np.interp(benchmark_reduced_nu, reduced_nu[order], qi_profile_percent[order])
    return float(np.sqrt(np.mean((model_at_benchmark - benchmark_qi_percent) ** 2)))


def main(warm_start=True):
    r"""
    Emergent :math:`Q/I` profile of the TM99 :math:`J=0 \to 1` scattering line at an inclined line
    of sight (:math:`\mu=0.1`), overlaid on the digitized TM99 (ApJ 516, 436) Fig. 10.
    """
    setup_logging()

    temperature_K = 6000.0
    epsilon = 1.0e-2
    mu_observer = 0.1
    n_mu_tm99 = 15

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition, epsilon, temperature_K)

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(), params.delta_v_thermal_cm_sm1, step_doppler=0.1
    )

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(1000e6, n_near_surface=100, n_interior=30),
        temperature_K=temperature_K,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    initial_state = load_warm_state(__file__, warm_start)
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
        ng_damping=0.5,
        ng_period=7,
        transfer_scheme="delo_linear",
        estimate_true_error=True,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu), initial_state=initial_state)
    save_warm_state(__file__, atmosphere)

    nu0 = transition.get_mean_transition_frequency_sm1()
    reduced_nu = reduced_frequency(nu, nu0, params.delta_v_thermal_cm_sm1)
    qi_profile_percent = 100.0 * emergent.Q / emergent.I

    logging.info("TM99 benchmark (mu = 0.1): epsilon = %.0e, delta2 = 0, no continuum, no field", epsilon)
    logging.info(
        "vertical optical thickness = %.1f, iterations = %d, residual = %.2e",
        float(atmosphere.tau_grid[-1]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )

    # TM99 Fig. 10 (delta2 = 0, mu = 0.1) digitized, blue wing mirrored onto the red.
    tm99_rf = np.array(np.arange(-5, 1e-4, 0.15))
    tm99_qi_dig = np.array(
        [
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
            0.02,
            0.03,
            0.09,
            0.16,
            0.29,
            0.43,
            0.62,
            0.77,
            0.85,
            0.73,
            0.43,
            -0.04,
            -0.62,
            -1.21,
            -1.77,
            -2.26,
            -2.68,
            -2.97,
            -3.19,
            -3.33,
            -3.38,
        ]
    )

    tm99_reduced_frequency_full2 = np.concatenate([tm99_rf, -tm99_rf[::-1]])
    tm99_qi_percent_full2 = np.concatenate([tm99_qi_dig, tm99_qi_dig[::-1]])
    rms_percent = benchmark_rms(reduced_nu, qi_profile_percent, tm99_reduced_frequency_full2, tm99_qi_percent_full2)
    rms = rms_percent / 100.0

    fig_qi, ax_qi = plt.subplots(figsize=(7, 5))

    # Emergent Q/I profile at mu = 0.1 (TM99 Fig. 10, delta2 = 0).
    ax_qi.axhline(0.0, color="k", linewidth=0.8)
    ax_qi.plot(
        tm99_reduced_frequency_full2,
        tm99_qi_percent_full2,
        linestyle="none",
        marker="x",
        markersize=5.5,
        markeredgewidth=1.3,
        color="k",
        label="TM99",
    )
    ax_qi.plot(reduced_nu, qi_profile_percent, "k-", label="SolRaT")
    ax_qi.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    ax_qi.set_ylabel(r"$100\,Q/I$")
    ax_qi.legend()
    fig_qi.tight_layout()
    print(
        f"TM99 Fig. 10, mu=0.1: RMS Q/I = {rms:.3e} "
        f"(iterations={atmosphere.iterations_used}, final residual={atmosphere.final_residual:.2e})"
    )
    return fig_qi


if __name__ == "__main__":
    main()
    plt.show()
