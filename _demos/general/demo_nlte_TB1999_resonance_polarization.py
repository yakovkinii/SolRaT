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
_MU_OBSERVER = 0.0  # tangential line of sight (mu = 0): emergent Q/I = surface source function (Table 4)
# TB1999 Table 4 reference (epsilon=1e-2, delta2=0), fully angle/space-resolved. The double-Gauss
# quadrature counts n_mu_quadrature // 2 points per hemisphere (same convention as TB1999's n_mu), so
# n_mu_quadrature = 30 matches their converged n_mu = 15
_TB1999_SURFACE_ALIGNMENT = 0.05666  # surface rho^2_0/rho^0_0
_TB1999_QI_PERCENT_TANGENTIAL = -6.132  # tangential (mu = 0) emergent line-center Q/I


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


def log_depth_grid(z_max_cm: float, n_depth: int, min_fraction: float = 1e-9) -> np.ndarray:
    r"""
    Height grid on ``[0, z_max_cm]`` with the depth *below the observer surface* logarithmically
    spaced, so the surface optical-depth decades are resolved (to match the log optical-depth axis
    of TB1999 Fig. 1 / Fig. 8). ``z[0]`` is the lower boundary, ``z[-1]`` the observer surface.
    ``min_fraction`` sets the thinnest top cell as a fraction of the slab: since the optical depth
    scales with geometric depth (constant N), the shallowest sampled optical depth from the surface
    is ``min_fraction`` times the total optical thickness, so a small value is needed to reach the
    ~1e-3 surface decades of the benchmark.
    """
    depth_below_surface = np.logspace(np.log10(z_max_cm * min_fraction), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth_below_surface)


def build_frequency_grid(transition, delta_v_thermal_cm_sm1: float) -> np.ndarray:
    r"""
    Frequency grid at ~2 points per Doppler width over +-4 Doppler widths (TB1999 note that two
    frequency points per Doppler width suffice for the isothermal Gaussian-profile benchmark).
    """
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
    step = 0.5 * delta_nu_D  # Use 0.25 for better match
    return np.arange(nu0 - 4.0 * delta_nu_D, nu0 + 4.0 * delta_nu_D + 0.5 * step, step)


def upper_level_alignment(atmosphere: NLTEStratifiedAtmosphere, upper_level_id: str) -> np.ndarray:
    r"""
    Fractional atomic alignment :math:`\rho^2_0 / \rho^0_0` of the upper level over the depth grid
    (TB1999 Fig. 1, solid line). The ratio is independent of the overall density-matrix
    normalization, so it compares directly with the paper.
    """
    sigma = []
    for rho in atmosphere.rho_grid:
        rho00 = np.real(rho(K=0, Q=0, level_id=upper_level_id))
        rho20 = np.real(rho(K=2, Q=0, level_id=upper_level_id))
        sigma.append(rho20 / rho00)
    return np.array(sigma)


def main():
    r"""
    TB1999 benchmark reproduction (for manual comparison against Trujillo Bueno & Manso Sainz 1999,
    ApJ 516, 436).

    Their standard resonance-line-polarization benchmark: a J=0 -> J=1 two-level atom in an
    isothermal, self-emitting, plane-parallel slab, no background continuum, no magnetic field
    (Hanle factor H^(2) = 1), no depolarizing collisions (delta^(2) = 0), and photon destruction
    probability epsilon (set through the parametrized collisional de-excitation rate). This demo uses
    epsilon = 1e-2, which converges far faster than TB1999's headline 1e-4 while still comparing to
    tabulated values (their Table 4).

    The plot is the upper-level alignment rho^2_0/rho^0_0 versus optical depth from the surface (TB1999
    Fig. 1 / Fig. 8, delta2=0), with a reference line at the tabulated surface value (Table 4). The log
    also reports the tangential (mu = 0) line-center Q/I against the Table 4 value: the mu = 0 emergent
    is the Eddington-Barbier limit I(0) = S(tau=0) (the surface source function), computed without
    integrating the ray. The frequency-resolved emergent Q/I profile (mu = 0.1, TB1999 Fig. 10) is a
    separate demo, demo_nlte_TB1999_resonance_polarization_mu01.py.

    Note on angular resolution: TB1999 Table 2/3 show the surface rho^2_0/rho^0_0 approaching its
    converged value only as the angular quadrature is refined. This code uses a double-Gauss mu rule
    (an independent Gauss-Legendre rule per hemisphere), which counts n_mu_quadrature // 2 points per
    hemisphere -- the same convention as TB1999's n_mu -- and, by putting the surface mu = 0 kink on a
    subinterval boundary, converges to the tabulated value instead of a low-biased one. n_mu_quadrature
    = 30 matches their converged n_mu = 15; coarser values fall short.

    The optical-depth grid is log-spaced in depth below the surface so it spans the surface decades
    (roughly 1e-3 up to the total optical thickness, here ~1e6, effectively semi-infinite).

    Convergence caveat: plain Lambda-iteration converges in ~1/epsilon iterations (TB1999 Figs. 2,
    4, 6); at epsilon = 1e-2 that is a few hundred iterations (Ng acceleration cuts it further).
    Starting from the isotropic LTE guess the alignment builds up from below toward the tabulated
    value; the residual and the achieved surface value are logged. A diagonal-operator ALI/SOR method
    (checklist A2) would converge faster still.
    """
    setup_logging()

    number_density_cm3 = 1.0e11  # constant absorber density; sets the total vertical optical thickness
    z_max_cm = 1000e5  # slab thickness [cm] (1000 km)
    n_depth = 80  # use 400 for better match

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    upper_level_id = transition.level_upper.level_id
    collisions.set_deexcitation_rate(transition.transition_id, c_ul_for_epsilon(_EPSILON, transition, _TEMPERATURE_K))

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=_TEMPERATURE_K
    )
    nu = build_frequency_grid(transition, params.delta_v_thermal_cm_sm1)
    line_center_index = int(np.argmin(np.abs(nu - transition.get_mean_transition_frequency_sm1())))

    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=log_depth_grid(z_max_cm, n_depth),
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
        los_theta=float(np.arccos(_MU_OBSERVER)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,  # Use 30 for better match
        n_phi_quadrature=3,
        max_iterations=1000,
        tolerance=1e-8,  # Use 1e-10 for better match
        ng_acceleration=True,
        ng_damping=0.7,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

    vertical_tau = atmosphere.tau_grid  # tau_grid is the vertical line optical depth (observer-independent)
    optical_depth_from_surface = vertical_tau[-1] - vertical_tau
    alignment = upper_level_alignment(atmosphere, upper_level_id)
    surface_alignment = alignment[-1]
    emergent_qi_percent = 100.0 * emergent.Q[line_center_index] / emergent.I[line_center_index]

    logging.info("TB1999 benchmark: epsilon = %.0e, delta2 = 0, no continuum, no field", _EPSILON)
    logging.info(
        "vertical optical thickness = %.1f, iterations = %d, residual = %.2e",
        float(vertical_tau[-1]),
        atmosphere.iterations_used,
        atmosphere.final_residual,
    )
    logging.info(
        "surface rho^2_0/rho^0_0 = %.5f  (TB1999: %.5f)", surface_alignment, _TB1999_SURFACE_ALIGNMENT
    )
    logging.info(
        "tangential (mu=0) line-center Q/I = %.3f %%  (TB1999 Table 4: %.3f %%)",
        emergent_qi_percent,
        _TB1999_QI_PERCENT_TANGENTIAL,
    )

    _, ax_alignment = plt.subplots(figsize=(7, 5))

    # Internal atomic alignment vs optical depth (TB1999 Fig. 1 / Fig. 8, delta2 = 0).
    ax_alignment.axhline(
        _TB1999_SURFACE_ALIGNMENT,
        color="k",
        linestyle="--",
        label=f"TB1999 surface value = {_TB1999_SURFACE_ALIGNMENT}",
    )
    ax_alignment.plot(optical_depth_from_surface[:-1], alignment[:-1], marker=".", label="SolRaT")
    ax_alignment.set_xscale("log")
    ax_alignment.set_xlabel(r"optical depth from surface  $\tau$")
    ax_alignment.set_ylabel(r"upper-level alignment  $\rho^2_0 / \rho^0_0$")
    ax_alignment.set_ylim(-0.02, 0.10)
    ax_alignment.set_title(rf"TB1999 Fig. 1 / Fig. 8 ($\epsilon = {_EPSILON:.0e}$, $\delta^2 = 0$)")
    ax_alignment.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
