import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from solrat.atom_model.model_registry import Models
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
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging

# Fe I 5434-like construction. The second-order (Paschen-Back) Zeeman signal is made visible at only
# a few kilogauss by using a high-spin term (S = 3/2, a 4P term) rather than a simple doublet: the
# larger intra-term coupling means modest mixing -- as in the Fe I 5434 line, whose 4P-like term
# levels are far apart yet mix appreciably by ~20 kG -- already reshapes the observed line by ~5 kG.
# The observed line is 4P_5/2 -> 4S_3/2 (upper J = 5/2, so it can still carry alignment for the
# scattering comparison). The 4P_5/2 level mixes with its 4P_3/2 and 4P_1/2 partners, kept a few
# Doppler widths away so the fine-structure branches stay cleanly resolved.
LOWER_ENERGY_CMM1 = 0.0  # 4S_3/2 lower level (single J)
UPPER_P52_ENERGY_CMM1 = 20_000.0  # 4P_5/2 (observed branch)
UPPER_P32_ENERGY_CMM1 = 20_001.0  # 4P_3/2 (satellite)
UPPER_P12_ENERGY_CMM1 = 20_001.6  # 4P_1/2 (satellite)
EINSTEIN_A_UL_SM1 = 1.0e7
ATOMIC_MASS_AMU = 56.0
TERM_SPIN = 1.5  # S = 3/2 (quartet terms): stronger J-mixing than a doublet
J_OBSERVED_UPPER = 2.5
J_OBSERVED_LOWER = 1.5
TEMPERATURE_K = 3000.0  # low T -> narrow thermal line so the Zeeman components are resolved
DELTA_V_TURBULENT_CM_SM1 = 0.0  # no extra broadening: keep the components sharp to compare positions
VOIGT_A = 0.01


def reference_frequency_and_wavelength():
    r"""
    Frequency and air reference wavelength of the observed 4P_5/2 -> 4S_3/2 branch.

    :return: tuple (nu0 [1/s], reference air wavelength [Angstrom]).
    """
    nu0 = energy_cmm1_to_frequency_sm1(UPPER_P52_ENERGY_CMM1 - LOWER_ENERGY_CMM1)
    return nu0, lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu0))


def build_multi_term_doublet(reference_lambda_A_air: float, lte: bool, j_constrained: bool = True):
    r"""
    The 4S_3/2 -> 4P multiplet as a multi-term atom. The full 4P term (J = 1/2, 3/2, 5/2) always
    enters the Hamiltonian, so the observed line carries the intra-term Paschen-Back J-mixing
    (second-order Zeeman). When ``j_constrained`` is True the radiative-transfer branch is restricted
    to the observed upper J = 5/2 line; when False the atom radiates all three fine-structure branches.

    :param reference_lambda_A_air: air reference wavelength [Angstrom].
    :param lte: if True use the LTE multi-term model (no scattering), else the NLTE one.
    :param j_constrained: activate the J constraints registered on the transition (isolate the J = 5/2
        branch). Without it all three fine-structure lines are radiated.
    :return: configured multi-term Model.
    """
    level_registry = MultiTermLevelRegistry()
    level_registry.register_level(beta="lower", L=0, S=TERM_SPIN, J=1.5, energy_cmm1=LOWER_ENERGY_CMM1)
    level_registry.register_level(beta="upper", L=1, S=TERM_SPIN, J=0.5, energy_cmm1=UPPER_P12_ENERGY_CMM1)
    level_registry.register_level(beta="upper", L=1, S=TERM_SPIN, J=1.5, energy_cmm1=UPPER_P32_ENERGY_CMM1)
    level_registry.register_level(beta="upper", L=1, S=TERM_SPIN, J=2.5, energy_cmm1=UPPER_P52_ENERGY_CMM1)
    level_registry.validate()

    transition_registry = MultiTermTransitionRegistry()
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="upper", L=1, S=TERM_SPIN),
        term_lower=level_registry.get_term(beta="lower", L=0, S=TERM_SPIN),
        einstein_a_ul_sm1=EINSTEIN_A_UL_SM1,
        lower_J_constraint=[J_OBSERVED_LOWER],
        upper_J_constraint=[J_OBSERVED_UPPER],
    )
    # j_constrained is what actually activates the J constraints above: without it the multi-term
    # atom radiates every J branch of the term (here all of 4P_1/2, 4P_3/2, 4P_5/2), and the RTE
    # constraints registered on the transition are ignored.
    config = MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=reference_lambda_A_air,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        j_constrained=j_constrained,
    )
    model = Models.multi_term_atom_lte() if lte else Models.multi_term_atom()
    return model.configure(config=config)


def build_multi_level_branch(reference_lambda_A_air: float):
    r"""
    The observed 4P_5/2 -> 4S_3/2 line as an isolated multi-level transition. Each J is an independent
    level, so the line is strictly linear in B (no intra-term J-mixing).

    :param reference_lambda_A_air: air reference wavelength [Angstrom].
    :return: configured multi-level Model.
    """
    lande_upper = 1.0 + (J_OBSERVED_UPPER * (J_OBSERVED_UPPER + 1) + TERM_SPIN * (TERM_SPIN + 1) - 1 * (1 + 1)) / (
        2 * J_OBSERVED_UPPER * (J_OBSERVED_UPPER + 1)
    )  # 4P_5/2 LS Lande factor (g = 8/5)
    level_registry = MultiLevelLevelRegistry()
    level_registry.register_level(alpha="lower", J=J_OBSERVED_LOWER, energy_cmm1=LOWER_ENERGY_CMM1, g=2.0)  # 4S_3/2
    level_registry.register_level(alpha="upper", J=J_OBSERVED_UPPER, energy_cmm1=UPPER_P52_ENERGY_CMM1, g=lande_upper)
    transition_registry = MultiLevelTransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="upper", J=J_OBSERVED_UPPER),
        level_lower=level_registry.get_level(alpha="lower", J=J_OBSERVED_LOWER),
        einstein_a_ul_sm1=EINSTEIN_A_UL_SM1,
    )
    config = MultiLevelAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        reference_lambda_A_air=reference_lambda_A_air,
        collisions=None,
    )
    return Models.multi_level_atom().configure(config=config)


def synthesize(model, nu: np.ndarray, angles: Angles, magnetic_field_gauss: float, anisotropic: bool) -> Stokes:
    r"""
    Constant-property-slab synthesis with either an isotropic (Planck) or an anisotropic
    (parametrized) radiation tensor.

    :param model: configured Model.
    :param nu: frequency grid [1/s].
    :param angles: geometry.
    :param magnetic_field_gauss: field strength [G].
    :param anisotropic: if True use the anisotropic radiation tensor (drives scattering polarization);
        if False use the isotropic Planck tensor (pure Zeeman line).
    :return: emergent Stokes vector.
    """
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=magnetic_field_gauss,
        temperature_K=TEMPERATURE_K,
        delta_v_turbulent_cm_sm1=DELTA_V_TURBULENT_CM_SM1,
        voigt_a=VOIGT_A,
    )
    # The LTE radiation tensor has no fill_* methods (the LTE SEE ignores the radiation field), so
    # only fill it for the NLTE models; the bare LTE tensor is passed through unchanged.
    radiation_tensor = model.RadiationTensor.from_model_config(model.config)
    if anisotropic and hasattr(radiation_tensor, "fill_NLTE_n_w_parametrized"):
        radiation_tensor = radiation_tensor.fill_NLTE_n_w_parametrized(h_arcsec=30)
    elif not anisotropic and hasattr(radiation_tensor, "fill_planck"):
        radiation_tensor = radiation_tensor.fill_planck(temperature_K=TEMPERATURE_K)
    atmosphere = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            line_delta_tau=0.3,
            continuum_delta_tau=1.0e-3,
            angles=angles,
            atmosphere_parameters=atmosphere_parameters,
        )
    )
    return atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))


def second_order_zeeman_figure(nu, nu0, reference_lambda_A_air, delta_nu_D):
    r"""
    Part (a): isolate second-order Zeeman. Three rows per field (columns): normalized Stokes I over
    the full multiplet, the same I zoomed to the observed-line core (reduced frequency in [-5, 5]),
    and V/max(I) over that same core. Each panel overlays three atomic descriptions of the observed
    line under an isotropic field, scanning B from the linear regime into the incomplete Paschen-Back
    regime: the multi-term atom with the radiative-transfer branch constrained to J = 5/2 (full
    intra-term J-mixing), the
    same multi-term atom without the constraint (all fine-structure branches radiated, showing the
    4P_3/2 and 4P_1/2 satellites the constraint removes), and the multi-level atom (strictly linear in
    B). With thin lines the fine-structure branches are spectrally resolved, so isolating one branch
    with the J constraint is physically meaningful. The constrained multi-term and the multi-level
    agree at low B and diverge as the field mixes the 4P term.

    :return: the matplotlib Figure.
    """
    angles = Angles(chi=0.0, theta=np.pi / 6, gamma=0.0, chi_B=0.0, theta_B=np.deg2rad(30.0))
    model_mt_constrained = build_multi_term_doublet(reference_lambda_A_air, lte=True, j_constrained=True)
    model_mt_full = build_multi_term_doublet(reference_lambda_A_air, lte=True, j_constrained=False)
    model_ml = build_multi_level_branch(reference_lambda_A_air)
    reduced_frequency = (nu - nu0) / delta_nu_D

    field_values_gauss = [500.0, 1000.0, 2000.0]
    zoom_limits = (-5.0, 5.0)  # reduced-frequency window on the observed-line core (satellites excluded)
    fig, axes = plt.subplots(3, len(field_values_gauss), figsize=(12, 10), sharey="row")
    worst_delta_v = 0.0
    for column, magnetic_field_gauss in enumerate(field_values_gauss):
        stokes_mt_full = synthesize(model_mt_full, nu, angles, magnetic_field_gauss, anisotropic=False)
        stokes_mt_constrained = synthesize(model_mt_constrained, nu, angles, magnetic_field_gauss, anisotropic=False)
        stokes_ml = synthesize(model_ml, nu, angles, magnetic_field_gauss, anisotropic=False)

        max_deltaV = np.max(
            np.abs(stokes_mt_constrained.V / np.max(stokes_mt_constrained.I) - stokes_ml.V / np.max(stokes_ml.I))
        )
        worst_delta_v = max(worst_delta_v, float(max_deltaV))

        ax_intensity_full, ax_intensity_zoom, ax_v_zoom = axes[0, column], axes[1, column], axes[2, column]
        for stokes, lw, color, linestyle in (
            (stokes_mt_full, 0.9, "#00FF00", "-"),
            (stokes_mt_constrained, 1.5, "#0000FF", "--"),
            (stokes_ml, 2.2, "k", (0, (1, 1))),
        ):
            intensity = stokes.I / np.max(stokes.I)
            v_over_imax = stokes.V / np.max(stokes.I)
            ax_intensity_full.plot(reduced_frequency, intensity, lw=lw, color=color, ls=linestyle)
            ax_intensity_zoom.plot(reduced_frequency, intensity, lw=lw, color=color, ls=linestyle)
            ax_v_zoom.plot(reduced_frequency, v_over_imax, lw=lw, color=color, ls=linestyle)
        ax_intensity_full.set_title(f"B = {magnetic_field_gauss:.0f} G")
        ax_intensity_zoom.set_xlim(zoom_limits)
        ax_v_zoom.set_xlim(zoom_limits)
        ax_v_zoom.set_xlabel(r"$(\nu - \nu_0)/\Delta\nu_D$")
        for ax in (ax_intensity_full, ax_intensity_zoom, ax_v_zoom):
            ax.axhline(0.0, color="0.7", lw=0.6)
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(r"$I\,/\,I_{\max}$ (full)")
    axes[1, 0].set_ylabel(r"$I\,/\,I_{\max}$ (zoom)")
    axes[2, 0].set_ylabel(r"$V\,/\,I_{\max}$ (zoom)")
    style_key = [
        Line2D([], [], color="#00FF00", lw=0.9, label="Multi-term, all branches"),
        Line2D([], [], color="#0000FF", lw=1.5, ls="--", label="Multi-term, $J$-constrained"),
        Line2D([], [], color="k", lw=2.2, ls=(0, (1, 1)), label="Multi-level (linear)"),
    ]
    axes[0, 0].legend(handles=style_key, fontsize=8, loc="best")
    fig.tight_layout()
    print(
        f"Second-order Zeeman (MT J-constrained vs ML): max|Delta V/I_max| = {worst_delta_v:.2e} "
        f"over B = {[int(b) for b in field_values_gauss]} G (should grow from ~0 at low field into "
        f"incomplete Paschen-Back)"
    )
    return fig


def nlte_scattering_figure(nu, nu0, reference_lambda_A_air, delta_nu_D):
    r"""
    Part (b): isolate NLTE scattering. At weak field and under an anisotropic radiation field, the
    multi-level atom develops a scattering-polarization Q/I from the self-consistent upper-level
    alignment, while the LTE multi-term atom (thermal populations, no alignment) does not.

    :return: the matplotlib Figure.
    """
    angles = Angles(chi=0.0, theta=np.pi / 3, gamma=0.0, chi_B=0.0, theta_B=0.0)
    magnetic_field_gauss = 50.0  # weak field: scattering, not Zeeman, dominates the linear polarization
    model_ml = build_multi_level_branch(reference_lambda_A_air)
    model_mt_lte = build_multi_term_doublet(reference_lambda_A_air, lte=True)
    reduced_frequency = (nu - nu0) / delta_nu_D

    stokes_ml = synthesize(model_ml, nu, angles, magnetic_field_gauss, anisotropic=True)
    stokes_mt_lte = synthesize(model_mt_lte, nu, angles, magnetic_field_gauss, anisotropic=True)
    qi_ml = 100.0 * stokes_ml.Q / stokes_ml.I
    qi_mt_lte = 100.0 * stokes_mt_lte.Q / stokes_mt_lte.I
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(reduced_frequency, qi_ml, lw=1.5, color="#1f77b4", label="Multi-level (NLTE scattering)")
    ax.plot(reduced_frequency, qi_mt_lte, lw=1.5, ls="--", color="#d62728", label="Multi-term (LTE, no scattering)")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$(\nu - \nu_0)/\Delta\nu_D$")
    ax.set_ylabel("$100\\,Q/I$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    print(
        f"Weak-field scattering Q/I: ML (NLTE) max|Q/I| = {np.max(np.abs(qi_ml)):.3e} %, "
        f"MT (LTE) max|Q/I| = {np.max(np.abs(qi_mt_lte)):.3e} % (MT-LTE should be ~0)"
    )
    return fig


def main():
    r"""
    Where the multi-term and multi-level descriptions diverge, decomposed into the two effects that a
    v1 SolRaT model carries but its counterpart does not (checklist A51). This is model-choice
    guidance, not a validation: neither model has both effects, so "more correct" is regime-dependent.

    Part (a) isolates the second-order Zeeman effect: the multi-term atom keeps the full intra-term
    Paschen-Back J-mixing (its radiative-transfer branch restricted to the observed J = 5/2 line),
    while the multi-level atom treats each J as an independent, strictly linear level. Under an
    isotropic field the two Stokes V/I profiles agree at low field and diverge as the field enters the
    incomplete-Paschen-Back regime.

    Part (b) isolates NLTE scattering: at weak field and anisotropic illumination the multi-level atom
    builds a self-consistent upper-level alignment and hence a scattering Q/I, which the LTE multi-term
    atom (thermal populations) does not reproduce.

    :return: a tuple ``(second_order_zeeman_figure, nlte_scattering_figure)`` (neither shown; the
        caller decides whether to display or save them; they are separate manuscript figures).
    """
    setup_logging()

    nu0, reference_lambda_A_air = reference_frequency_and_wavelength()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.5,
        upper_wavelength_A=reference_lambda_A_air + 0.5,
        step_A=2e-4,
    )
    scale_model = build_multi_level_branch(reference_lambda_A_air)
    delta_v_thermal_cm_sm1 = scale_model.AtmosphereParameters(
        model_config=scale_model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K,
        delta_v_turbulent_cm_sm1=DELTA_V_TURBULENT_CM_SM1, voigt_a=VOIGT_A,
    ).delta_v_thermal_cm_sm1  # fmt: skip
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1

    second_order_zeeman = second_order_zeeman_figure(nu, nu0, reference_lambda_A_air, delta_nu_D)
    nlte_scattering = nlte_scattering_figure(nu, nu0, reference_lambda_A_air, delta_nu_D)
    return second_order_zeeman, nlte_scattering


if __name__ == "__main__":
    main()
    plt.show()
