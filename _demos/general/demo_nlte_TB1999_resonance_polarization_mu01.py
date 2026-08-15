import logging

import matplotlib.pyplot as plt
import numpy as np
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

TEMPERATURE_K = 6000.0
EPSILON = 1.0e-2  # photon destruction probability
MU_OBSERVER = 0.1  # inclined line of sight


def c_ul_for_epsilon(epsilon: float, transition, temperature_K: float) -> float:
    r"""
    Collisional de-excitation rate :math:`C_{ul}` [1/s] that yields a two-level-atom photon
    destruction probability ``epsilon``:
    :math:`\epsilon = C_{ul}(1 - e^{-h\nu_0/kT}) / (A_{ul} + C_{ul}(1 - e^{-h\nu_0/kT}))`, so
    :math:`C_{ul} = \frac{\epsilon}{1-\epsilon}\, A_{ul} / (1 - e^{-h\nu_0/kT})`.

    Reference: two-level-atom destruction probability (e.g. Mihalas 1978; TB1999 Sec. 2).
    """
    assert 0.0 < epsilon < 1.0, "epsilon must be in (0, 1)."
    delta_e_erg = (transition.level_upper.energy_cmm1 - transition.level_lower.energy_cmm1) * h_erg_s * c_cm_sm1
    stimulated_correction = 1.0 - exp(-delta_e_erg / (kB_erg_Km1 * temperature_K))
    return epsilon / (1.0 - epsilon) * transition.einstein_a_ul / stimulated_correction


def surface_refined_depth_grid(
    z_max_cm: float, n_surface: int, n_deep: int, surface_fraction: float = 1e-3, min_fraction: float = 1e-7
) -> np.ndarray:
    r"""
    Depth grid concentrated near the observer surface (where the inclined-ray line core forms), with
    a sparse thermalized interior. ``z[0]`` is the lower boundary, ``z[-1]`` the observer surface.

    :param z_max_cm: slab thickness [cm].
    :param n_surface: number of logarithmically packed surface points.
    :param n_deep: number of sparse interior points.
    :param surface_fraction: depth (fraction of the slab) separating the packed surface from the interior.
    :param min_fraction: thinnest top cell as a fraction of the slab.
    :return: sorted height grid [cm].
    """
    surface = np.logspace(np.log10(min_fraction), np.log10(surface_fraction), n_surface, endpoint=False)
    deep = np.logspace(np.log10(surface_fraction), 0.0, n_deep)
    depth_below_surface = z_max_cm * np.concatenate([surface, deep])
    return np.sort(z_max_cm - depth_below_surface)


def build_frequency_grid(transition, delta_v_thermal_cm_sm1: float) -> np.ndarray:
    r"""
    Frequency grid covering the line for the synthesis.

    :param transition: the radiative transition.
    :param delta_v_thermal_cm_sm1: thermal+turbulent Doppler velocity [cm/s].
    :return: frequency grid [1/s].
    """
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
    step = 0.1 * delta_nu_D
    return np.arange(nu0 - 4.0 * delta_nu_D, nu0 + 4.0 * delta_nu_D + 0.5 * step, step)


def main():
    r"""
    Emergent :math:`Q/I` profile of the TB1999 :math:`J=0 \to 1` scattering line at an inclined line
    of sight (:math:`\mu=0.1`), overlaid on the digitized TB1999 (ApJ 516, 436) Fig. 10.

    :return: matplotlib Figure.
    """
    setup_logging()

    number_density_cm3 = 1.0e11
    z_max_cm = 1000e5
    n_surface = 50
    n_deep = 20

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate(transition.transition_id, c_ul_for_epsilon(EPSILON, transition, TEMPERATURE_K))

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
    )
    nu = build_frequency_grid(transition, params.delta_v_thermal_cm_sm1)

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=surface_refined_depth_grid(z_max_cm, n_surface, n_deep),
        temperature_K=TEMPERATURE_K,
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
        los_theta=float(np.arccos(MU_OBSERVER)),
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
    reduced_frequency = (nu - nu0) / (nu0 * params.delta_v_thermal_cm_sm1 / c_cm_sm1)
    qi_profile_percent = 100.0 * emergent.Q / emergent.I

    logging.info("TB1999 benchmark (mu = 0.1): epsilon = %.0e, delta2 = 0, no continuum, no field", EPSILON)
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
    ])  # fmt: skip
    tb_qi_percent = np.array([
        0.0, -0.00226, -0.00792, -0.00792, -0.00226, -0.01075, 0.01188, 0.0543, 0.13348, 0.28337,
        0.50113, 0.77828, 0.86312, 0.6086, 0.04581, -0.65554, -1.39367, -1.93665, -2.29016, -2.7681,
        -3.01697, -3.13292, -3.25735, -3.33371, -3.40158,
    ])  # fmt: skip
    tb_reduced_frequency_full = np.concatenate([tb_reduced_frequency, -tb_reduced_frequency[::-1]])
    tb_qi_percent_full = np.concatenate([tb_qi_percent, tb_qi_percent[::-1]])

    fig_qi, ax_qi = plt.subplots(figsize=(7, 5))

    # Emergent Q/I profile at mu = 0.1 (TB1999 Fig. 10, delta2 = 0).
    ax_qi.axhline(0.0, color="k", linewidth=0.8)
    ax_qi.plot(
        tb_reduced_frequency_full, tb_qi_percent_full, linestyle="none", marker="x", color="k",
        label="TB1999 Fig. 10 (digitized)",
    )  # fmt: skip
    ax_qi.plot(reduced_frequency, qi_profile_percent, marker=".", label=r"SolRaT ($\mu = 0.1$)")
    ax_qi.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    ax_qi.set_ylabel(r"$100\,Q/I$")
    ax_qi.legend()
    fig_qi.tight_layout()
    line_center_index = int(np.argmin(np.abs(reduced_frequency)))
    print(
        f"TB1999 Fig. 10 (mu=0.1): iterations = {atmosphere.iterations_used}, final residual = "
        f"{atmosphere.final_residual:.2e}, line-center 100 Q/I = {qi_profile_percent[line_center_index]:.3f} "
        f"(TBD: re-run at better convergence for the final figure -- see final-review flag FR9)"
    )
    return fig_qi


if __name__ == "__main__":
    main()
    plt.show()
