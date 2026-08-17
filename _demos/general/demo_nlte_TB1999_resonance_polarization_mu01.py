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
from solrat.atom_model.shared.utility.functions import (
    frequencies_around_line_sm1,
    height_grid_refined_at_observer_surface,
    reduced_frequency,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


def main():
    r"""
    Emergent :math:`Q/I` profile of the TB1999 :math:`J=0 \to 1` scattering line at an inclined line
    of sight (:math:`\mu=0.1`), overlaid on the digitized TB1999 (ApJ 516, 436) Fig. 10.
    """
    setup_logging()

    temperature_K = 6000.0
    epsilon = 1.0e-2
    mu_observer = 0.1

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition, epsilon, temperature_K)

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu = frequencies_around_line_sm1(transition.get_mean_transition_frequency_sm1(), params.delta_v_thermal_cm_sm1)

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_grid_refined_at_observer_surface(1000e5, n_near_surface=50, n_interior=20),
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
        los_theta=float(np.arccos(mu_observer)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=1000,
        tolerance=1e-10,
        ng_acceleration=True,
        ng_damping=0.7,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

    nu0 = transition.get_mean_transition_frequency_sm1()
    reduced_nu = reduced_frequency(nu, nu0, params.delta_v_thermal_cm_sm1)
    qi_profile_percent = 100.0 * emergent.Q / emergent.I

    logging.info("TB1999 benchmark (mu = 0.1): epsilon = %.0e, delta2 = 0, no continuum, no field", epsilon)
    logging.info(
        "vertical optical thickness = %.1f, iterations = %d, residual = %.2e",
        float(atmosphere.tau_grid[-1]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )

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

    fig_qi, ax_qi = plt.subplots(figsize=(7, 5))

    # Emergent Q/I profile at mu = 0.1 (TB1999 Fig. 10, delta2 = 0).
    ax_qi.axhline(0.0, color="k", linewidth=0.8)
    ax_qi.plot(
        tb_reduced_frequency_full,
        tb_qi_percent_full,
        linestyle="none",
        marker="x",
        color="k",
        label="TB1999 Fig. 10 (digitized)",
    )
    ax_qi.plot(reduced_nu, qi_profile_percent, marker=".", label=r"SolRaT ($\mu = 0.1$)")
    ax_qi.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    ax_qi.set_ylabel(r"$100\,Q/I$")
    ax_qi.legend()
    fig_qi.tight_layout()
    line_center_index = int(np.argmin(np.abs(reduced_nu)))
    print(
        f"TB1999 Fig. 10 (mu=0.1): iterations = {atmosphere.iterations_used}, final residual = "
        f"{atmosphere.final_residual:.2e}, line-center 100 Q/I = {qi_profile_percent[line_center_index]:.3f} "
        f"(TBD: re-run at better convergence for the final figure -- see final-review flag FR9)"
    )
    return fig_qi


if __name__ == "__main__":
    main()
    plt.show()
