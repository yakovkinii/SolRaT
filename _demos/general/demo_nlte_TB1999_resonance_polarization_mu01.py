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

_TEMPERATURE_K = 6000.0  # isothermal slab
_EPSILON = 1.0e-2  # TB1999 photon destruction probability; 1e-2 converges far faster than their 1e-4
_MU_OBSERVER = 0.1  # line-of-sight direction cosine mu = 0.1 (TB1999 Fig. 10 emergent Q/I profile)


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
    Depth grid concentrated near the observer surface, for an inclined line of sight.

    ``n_surface`` points are packed logarithmically in the surface layers (depth below the surface
    ``min_fraction`` .. ``surface_fraction`` of the slab -- the low optical depths where an inclined
    ray forms its line core), and ``n_deep`` sparse points cover the thermalized interior
    (``surface_fraction`` .. 1, effectively semi-infinite). This resolves the inclined-ray DELO steps
    at line center, whose per-cell optical depth is amplified by 1/mu (Delta tau_LOS = Delta
    tau_vertical / mu) and which a uniform log grid under-resolves (a too-shallow line core). Since the
    optical depth scales with geometric depth (constant N), a fraction of the slab maps to that same
    fraction of the total optical thickness. ``z[0]`` is the lower boundary, ``z[-1]`` the surface.
    """
    surface = np.logspace(np.log10(min_fraction), np.log10(surface_fraction), n_surface, endpoint=False)
    deep = np.logspace(np.log10(surface_fraction), 0.0, n_deep)
    depth_below_surface = z_max_cm * np.concatenate([surface, deep])
    return np.sort(z_max_cm - depth_below_surface)


def build_frequency_grid(transition, delta_v_thermal_cm_sm1: float) -> np.ndarray:
    r"""
    Frequency grid at ~2 points per Doppler width over +-4 Doppler widths (TB1999 note that two
    frequency points per Doppler width suffice for the isothermal Gaussian-profile benchmark).
    """
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
    step = 0.1 * delta_nu_D
    return np.arange(nu0 - 4.0 * delta_nu_D, nu0 + 4.0 * delta_nu_D + 0.5 * step, step)


def main():
    r"""
    TB1999 benchmark reproduction, emergent Q/I profile at mu = 0.1 (for manual comparison against
    Trujillo Bueno & Manso Sainz 1999, ApJ 516, 436, Fig. 10, delta2 = 0).

    Same slab as demo_nlte_TB1999_resonance_polarization.py (a J=0 -> J=1 two-level atom in an
    isothermal, self-emitting, plane-parallel slab; no continuum, no field, no depolarizing collisions,
    epsilon = 1e-2), but instead of the tangential (mu = 0) surface source function this observes along
    an inclined line of sight, mu = 0.1, and integrates the ray through the atmosphere. That is what
    gives the emergent Q/I its frequency structure: each frequency probes a different optical depth
    (line center is opaque and samples the shallow, weakly aligned surface layers; the wings are
    transparent and sample deeper, more aligned layers), so Q/I(nu) is a profile rather than the
    flat surface-source-function value of the tangential case. Compare its shape and amplitude with
    TB1999 Fig. 10 (delta2 = 0 curve).

    The internal radiation field (hence the atomic alignment) is solved with the same double-Gauss mu
    quadrature and log depth grid as the tangential demo; only the observer line of sight differs.
    """
    setup_logging()

    number_density_cm3 = 1.0e11  # constant absorber density; sets the total vertical optical thickness
    z_max_cm = 1000e5  # slab thickness [cm] (1000 km)
    # Depth points concentrated in the surface layers (where the mu = 0.1 line core forms) and sparse
    # in the thermalized interior. Bump n_surface if the line core is still shallow vs TB1999 Fig. 10.
    n_surface = 80   # Use 500 for better match
    n_deep = 30  # Use 60 for better match

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate(transition.transition_id, c_ul_for_epsilon(_EPSILON, transition, _TEMPERATURE_K))

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=_TEMPERATURE_K
    )
    nu = build_frequency_grid(transition, params.delta_v_thermal_cm_sm1)

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=surface_refined_depth_grid(z_max_cm, n_surface, n_deep),
        temperature_K=_TEMPERATURE_K,
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
        los_theta=float(np.arccos(_MU_OBSERVER)),  # inclined line of sight, mu = 0.1
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10, # Use 50 for better match
        n_phi_quadrature=3,
        max_iterations=1000,
        tolerance=1e-8, # Use 1e-10 for better match
        ng_acceleration=True,  # Ng extrapolation of the rho iterates to cut the iteration count
        ng_damping=0.7,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

    nu0 = transition.get_mean_transition_frequency_sm1()
    reduced_frequency = (nu - nu0) / (nu0 * params.delta_v_thermal_cm_sm1 / c_cm_sm1)  # (nu - nu0)/Delta nu_D
    qi_profile_percent = 100.0 * emergent.Q / emergent.I  # emergent (mu = 0.1) Q/I profile

    logging.info("TB1999 benchmark (mu = 0.1): epsilon = %.0e, delta2 = 0, no continuum, no field", _EPSILON)
    logging.info(
        "vertical optical thickness = %.1f, iterations = %d, residual = %.2e",
        float(atmosphere.tau_grid[-1]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )

    # TB1999 Fig. 10 (delta2 = 0, mu = 0.1), digitized: reduced frequency (nu - nu0)/Delta nu_D and
    # 100 Q/I. Only the blue wing was digitized; the static isothermal profile is symmetric about line
    # center, so mirror it onto the red wing for a full-range comparison.
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

    _, ax_qi = plt.subplots(figsize=(7, 5))

    # Emergent Q/I profile at mu = 0.1 (TB1999 Fig. 10, delta2 = 0).
    ax_qi.axhline(0.0, color="k", linewidth=0.8)
    ax_qi.plot(
        tb_reduced_frequency_full, tb_qi_percent_full, linestyle="none", marker="x", color="k",
        label="TB1999 Fig. 10 (digitized)",
    )  # fmt: skip
    ax_qi.plot(reduced_frequency, qi_profile_percent, marker=".", label=r"SolRaT ($\mu = 0.1$)")
    ax_qi.set_xlabel(r"$(\nu - \nu_0)\,/\,\Delta\nu_D$")
    ax_qi.set_ylabel(r"$100\,Q/I$")
    ax_qi.set_title(rf"TB1999 Fig. 10 emergent Q/I ($\mu = 0.1$, $\delta^2 = 0$, $\epsilon = {_EPSILON:.0e}$)")
    ax_qi.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
