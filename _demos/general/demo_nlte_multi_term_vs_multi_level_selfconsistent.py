import logging

import matplotlib.pyplot as plt
import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry as MultiLevelLevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.transition_registry import (
    TransitionRegistry as MultiLevelTransitionRegistry,
)
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry as MultiTermLevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import (
    TransitionRegistry as MultiTermTransitionRegistry,
)
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequencies_around_line_sm1,
    frequency_sm1_to_lambda_A,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging

UPPER_ENERGY_CMM1 = 20_000.0  # ~5000 A
EINSTEIN_A_UL_SM1 = 1.0e7
ATOMIC_MASS_AMU = 56.0


def reference_wavelength_A_air():
    r"""
    Air reference wavelength of the shared ``UPPER_ENERGY_CMM1`` line.
    """
    return lambda_vacuum_to_air(frequency_sm1_to_lambda_A(energy_cmm1_to_frequency_sm1(UPPER_ENERGY_CMM1)))


def build_multi_term_model(reference_lambda_A_air, collisions=None):
    r"""
    The :math:`^1S_0 \to {}^1P_1` line (:math:`S=0`, one :math:`J` per term) as a multi-term atom.
    With a single :math:`J` per term there is no intra-term Paschen-Back :math:`J`-mixing, so the
    multi-term description reduces exactly to the multi-level one -- which lets a multi-term run be
    checked against the multi-level path bit for bit.
    """
    level_registry = MultiTermLevelRegistry()
    level_registry.register_level(beta="lower", L=0, S=0, J=0, energy_cmm1=0.0)
    level_registry.register_level(beta="upper", L=1, S=0, J=1, energy_cmm1=UPPER_ENERGY_CMM1)
    level_registry.validate()

    transition_registry = MultiTermTransitionRegistry()
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="upper", L=1, S=0),
        term_lower=level_registry.get_term(beta="lower", L=0, S=0),
        einstein_a_ul_sm1=EINSTEIN_A_UL_SM1,
    )
    config = MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=reference_lambda_A_air,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        collisions=collisions,
    )
    return Models.multi_term_atom().configure(config=config)


def build_multi_level_model(reference_lambda_A_air, collisions=None):
    r"""
    The same :math:`J=0 \to 1` resonance line as a multi-level atom (upper Lande factor
    :math:`g_u=1`), matching :func:`build_multi_term_model` in energy, Einstein coefficient, and mass.
    """
    level_registry = MultiLevelLevelRegistry()
    level_registry.register_level(alpha="lower", J=0, energy_cmm1=0.0, g=1.0)
    level_registry.register_level(alpha="upper", J=1, energy_cmm1=UPPER_ENERGY_CMM1, g=1.0)

    transition_registry = MultiLevelTransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="upper", J=1),
        level_lower=level_registry.get_level(alpha="lower", J=0),
        einstein_a_ul_sm1=EINSTEIN_A_UL_SM1,
    )
    config = MultiLevelAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        reference_lambda_A_air=reference_lambda_A_air,
        collisions=collisions,
    )
    return Models.multi_level_atom().configure(config=config)


def log_depth_grid(z_max_cm, n_depth, min_fraction=1e-9):
    r"""
    Height grid with the depth below the observer surface logarithmically spaced. ``z[0]`` is the
    lower boundary (deep), ``z[-1]`` the observer surface (optical depth :math:`\to 0`).
    """
    depth_below_surface = np.logspace(np.log10(z_max_cm * min_fraction), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth_below_surface)


def run_self_consistent(model, nu, temperature_K, number_density_cm3, mu_observer):
    r"""
    Solve the self-consistent height-stratified scattering problem for ``model`` and return the
    converged atmosphere and the emergent Stokes vector. Everything except the atomic description is
    identical between the two calls; the photon-destruction is carried by the model's collisions.
    """
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=log_depth_grid(1000e5, 80),
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
        los_theta=float(np.arccos(mu_observer)),
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=1000,
        tolerance=1e-8,
        ng_acceleration=True,
        ng_damping=0.7,
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    return atmosphere, emergent


def multi_term_alignment(atmosphere, term_id):
    r"""
    Fractional alignment :math:`\rho^2_0/\rho^0_0` of the upper term (:math:`J=J'=1`) over depth.
    """
    out = []
    for rho in atmosphere.rho_grid:
        # positional (K, Q, J, Jʹ, term_id) to avoid depending on the unicode Jʹ keyword name
        rho00 = np.real(rho(0, 0, 1.0, 1.0, term_id))
        rho20 = np.real(rho(2, 0, 1.0, 1.0, term_id))
        out.append(rho20 / rho00)
    return np.array(out)


def multi_level_alignment(atmosphere, level_id):
    r"""
    Fractional alignment :math:`\rho^2_0/\rho^0_0` of the upper level over depth.
    """
    out = []
    for rho in atmosphere.rho_grid:
        rho00 = np.real(rho(K=0, Q=0, level_id=level_id))
        rho20 = np.real(rho(K=2, Q=0, level_id=level_id))
        out.append(rho20 / rho00)
    return np.array(out)


def main():
    r"""
    Validate the multi-term self-consistent NLTE path on an :math:`S=0` line
    (:math:`^1S_0 \to {}^1P_1` / :math:`J=0 \to 1`), two ways at once.

    Both a multi-term and a multi-level atom are built for the same line, given the same
    photon-destruction probability :math:`\epsilon=10^{-2}` through the parametrized collisional
    de-excitation rate, and run through the same self-consistent height-stratified scattering
    atmosphere under identical conditions -- the setup of Trujillo Bueno & Manso Sainz (1999, TM99).

    * Direct benchmark: the multi-term surface alignment :math:`\rho^2_0/\rho^0_0` and the tangential
      (:math:`\mu=0`) line-center :math:`Q/I` are compared with the tabulated TM99 values.
    * Reduction cross-check: because the term carries a single :math:`J`, the two descriptions are
      formally identical, so the multi-term and multi-level solutions must agree to numerical
      precision. This anchors the multi-term run to the (independently TM99-validated) multi-level
      solver. The Paschen-Back machinery carried by the same multi-term SEE/RTE is validated
      separately (Zeeman pattern, Unno-Rachkovsky, and the Hazel2 He\,I D3 comparison).
    """
    setup_logging()

    temperature_K = 6000.0
    number_density_cm3 = 1.0e11
    epsilon = 1.0e-2
    mu_observer = 0.0  # tangential, to compare the line-center Q/I with TM99 Table 4
    tm99_surface_alignment = 0.05666  # TM99 Table 4
    tm99_qi_percent_tangential = -6.132  # TM99 Table 4

    reference_lambda_A_air = reference_wavelength_A_air()

    collisions_mt = ParametrizedCollisions()
    collisions_ml = ParametrizedCollisions()
    model_mt = build_multi_term_model(reference_lambda_A_air, collisions=collisions_mt)
    model_ml = build_multi_level_model(reference_lambda_A_air, collisions=collisions_ml)

    mt_transition = next(iter(model_mt.config.transition_registry.transitions.values()))
    ml_transition = next(iter(model_ml.config.transition_registry.transitions.values()))
    # Multi-term: spread a single multiplet epsilon over the (one, here) fine-structure component.
    # Multi-level: epsilon is set per transition (a single component). For a one-J-per-term line both
    # give the same C_ul, so the two self-consistent solutions must still match to machine precision.
    collisions_mt.fill_deexcitation_from_epsilon(mt_transition, epsilon, temperature_K)
    collisions_ml.set_deexcitation_rate_from_epsilon(ml_transition, epsilon, temperature_K)
    upper_term_id = mt_transition.term_upper.term_id
    upper_level_id = ml_transition.level_upper.level_id

    nu0 = mt_transition.get_mean_transition_frequency_sm1()
    params_mt = model_mt.AtmosphereParameters(
        model_config=model_mt.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu = frequencies_around_line_sm1(nu0, params_mt.delta_v_thermal_cm_sm1, step_doppler=0.5)
    line_center_index = int(np.argmin(np.abs(nu - nu0)))
    reduced_frequency = (nu - nu0) / (nu0 * params_mt.delta_v_thermal_cm_sm1 / c_cm_sm1)

    atmosphere_mt, emergent_mt = run_self_consistent(model_mt, nu, temperature_K, number_density_cm3, mu_observer)
    atmosphere_ml, emergent_ml = run_self_consistent(model_ml, nu, temperature_K, number_density_cm3, mu_observer)

    optical_depth_from_surface = atmosphere_mt.tau_grid[-1] - atmosphere_mt.tau_grid
    alignment_mt = multi_term_alignment(atmosphere_mt, upper_term_id)
    alignment_ml = multi_level_alignment(atmosphere_ml, upper_level_id)
    surface_alignment_mt = alignment_mt[-1]

    max_alignment_delta = float(np.max(np.abs(alignment_mt - alignment_ml)))
    max_stokes_delta = max(
        float(np.max(np.abs(getattr(emergent_mt, s) - getattr(emergent_ml, s)))) for s in ("I", "Q", "U", "V")
    )
    qi_mt = 100.0 * emergent_mt.Q[line_center_index] / emergent_mt.I[line_center_index]
    qi_ml = 100.0 * emergent_ml.Q[line_center_index] / emergent_ml.I[line_center_index]

    logging.info(
        "multi-term:  iterations = %d, residual = %.2e, tau_total = %.1f",
        atmosphere_mt.iterations_used,
        atmosphere_mt.final_residual,
        float(atmosphere_mt.tau_grid[-1]),
    )
    logging.info(
        "surface rho^2_0/rho^0_0 (multi-term) = %.5f  (TM99: %.5f)",
        surface_alignment_mt,
        tm99_surface_alignment,
    )

    fig, (ax_align, ax_qi) = plt.subplots(1, 2, figsize=(11, 5))
    ax_align.axhline(tm99_surface_alignment, color="k", ls="--", label=f"TM99 surface = {tm99_surface_alignment}")
    ax_align.plot(optical_depth_from_surface[:-1], alignment_mt[:-1], lw=1.4, color="#1f77b4", label="multi-term")
    ax_align.plot(
        optical_depth_from_surface[:-1], alignment_ml[:-1], lw=3.0, ls=(0, (1, 1)), color="#d62728", label="multi-level"
    )
    ax_align.set_xscale("log")
    ax_align.set_xlabel(r"optical depth from surface  $\tau$")
    ax_align.set_ylabel(r"upper-level alignment  $\rho^2_0 / \rho^0_0$")
    ax_align.set_ylim(-0.02, 0.10)
    ax_align.legend()

    ax_qi.plot(reduced_frequency, 100.0 * emergent_mt.Q / emergent_mt.I, lw=1.4, color="#1f77b4", label="multi-term")
    ax_qi.plot(
        reduced_frequency,
        100.0 * emergent_ml.Q / emergent_ml.I,
        lw=3.0,
        ls=(0, (1, 1)),
        color="#d62728",
        label="multi-level",
    )
    ax_qi.set_xlabel(r"reduced frequency  $v = (\nu-\nu_0)/\Delta\nu_D$")
    ax_qi.set_ylabel(r"emergent $Q/I$ (\%), $\mu=0$")
    ax_qi.legend()
    fig.suptitle(
        rf"Self-consistent NLTE ($\epsilon=10^{{-2}}$): multi-term vs TM99 and vs multi-level "
        rf"(max$|\Delta\rho^2_0/\rho^0_0|={max_alignment_delta:.1e}$)"
    )
    fig.tight_layout()

    print(
        f"Multi-term self-consistent NLTE (S=0 line, epsilon={epsilon:.0e}): "
        f"surface rho^2_0/rho^0_0 = {surface_alignment_mt:.5f} "
        f"(TM99 {tm99_surface_alignment:.5f}, "
        f"rel err {abs(surface_alignment_mt / tm99_surface_alignment - 1.0):.1%}); "
        f"tangential Q/I = {qi_mt:.3f}% (TM99 {tm99_qi_percent_tangential:.3f}%). "
        f"Multi-term vs multi-level: max|Delta rho^2_0/rho^0_0| = {max_alignment_delta:.2e}, "
        f"max|Delta Stokes| = {max_stokes_delta:.2e}, Q/I {qi_mt:.4f}% vs {qi_ml:.4f}% "
        f"(both should be ~machine precision)."
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
