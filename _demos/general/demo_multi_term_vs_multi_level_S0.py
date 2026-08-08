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

UPPER_ENERGY_CMM1 = 20_000.0  # ~5000 A transition, inside the n(lambda)/w(lambda) fit range
EINSTEIN_A_UL_SM1 = 1.0e7
ATOMIC_MASS_AMU = 56.0


def reference_frequency_and_wavelength():
    r"""
    Transition frequency and air reference wavelength of the shared ``UPPER_ENERGY_CMM1`` line.

    :return: tuple (nu0 [1/s], reference air wavelength [Angstrom]).
    """
    nu0 = energy_cmm1_to_frequency_sm1(UPPER_ENERGY_CMM1)
    return nu0, lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu0))


def build_multi_term_normal_triplet(reference_lambda_A_air: float):
    r"""
    Build the normal Zeeman triplet ^1S_0 -> ^1P_1 (S = 0, one J per term, Lande g_u = 1) as a
    multi-term atom. With a single J per term there is no intra-term Paschen-Back J-mixing, so the
    multi-term description reduces exactly to the multi-level one.

    :param reference_lambda_A_air: air reference wavelength of the line [Angstrom].
    :return: configured multi-term Model.
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
    )
    return Models.multi_term_atom().configure(config=config)


def build_multi_level_normal_triplet(reference_lambda_A_air: float):
    r"""
    Build the same line as a multi-level atom: a J = 0 -> J = 1 resonance transition with upper-level
    Lande factor g_u = 1 (the ^1P_1 value), matching :func:`build_multi_term_normal_triplet` in
    energy, Einstein coefficient, and mass.

    :param reference_lambda_A_air: air reference wavelength of the line [Angstrom].
    :return: configured multi-level Model.
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
        collisions=None,
    )
    return Models.multi_level_atom().configure(config=config)


def synthesize(model, nu: np.ndarray, angles: Angles, magnetic_field_gauss: float) -> Stokes:
    r"""
    Prescribed-J constant-property slab synthesis for either atom model, with the same anisotropic
    radiation tensor and atmosphere for both so that only the atom description differs.

    :param model: configured multi-term or multi-level Model.
    :param nu: frequency grid [1/s].
    :param angles: observation and magnetic-field geometry.
    :param magnetic_field_gauss: magnetic field strength [G].
    :return: emergent Stokes vector.
    """
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=magnetic_field_gauss,
        temperature_K=6000.0,
        delta_v_turbulent_cm_sm1=2.0e5,
        voigt_a=0.05,
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(h_arcsec=30)
    atmosphere = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            line_delta_tau=1.0,
            continuum_delta_tau=1.0e-3,
            angles=angles,
            atmosphere_parameters=atmosphere_parameters,
        )
    )
    return atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))


def main():
    r"""
    Multi-term vs multi-level agreement on an S = 0 line (^1S_0 -> ^1P_1 / J = 0 -> J = 1).

    The two pipelines are run through the same prescribed-J constant-property slab with an identical
    anisotropic radiation tensor, geometry, and atmosphere; only the atom model differs. Because the
    term has a single J, the multi-term atom carries no intra-term Paschen-Back J-mixing, so the two
    descriptions are formally identical and the emergent Stokes profiles must agree to numerical
    precision even at kilogauss fields. This is the agreement baseline for the multi-term / multi-level
    comparison (the counterpart to the fine-structure divergence demo).

    :return: the matplotlib Figure overlaying the multi-term (thin solid) and multi-level (thick
        short-dotted) Stokes profiles (not shown; the caller decides whether to display or save it).
    """
    setup_logging()

    magnetic_field_gauss = 1200.0
    angles = Angles(chi=0.0, theta=np.pi / 4, gamma=0.0, chi_B=np.deg2rad(30.0), theta_B=np.deg2rad(60.0))

    nu0, reference_lambda_A_air = reference_frequency_and_wavelength()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.35,
        upper_wavelength_A=reference_lambda_A_air + 0.35,
        step_A=1e-3,
    )

    model_mt = build_multi_term_normal_triplet(reference_lambda_A_air)
    model_ml = build_multi_level_normal_triplet(reference_lambda_A_air)
    stokes_mt = synthesize(model_mt, nu, angles, magnetic_field_gauss)
    stokes_ml = synthesize(model_ml, nu, angles, magnetic_field_gauss)

    delta_v_thermal_cm_sm1 = model_mt.AtmosphereParameters(
        model_config=model_mt.config, magnetic_field_gauss=magnetic_field_gauss, temperature_K=6000.0,
        delta_v_turbulent_cm_sm1=2.0e5, voigt_a=0.05,
    ).delta_v_thermal_cm_sm1  # fmt: skip
    reduced_frequency = (nu - nu0) / (nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1)  # (nu - nu0)/Delta nu_D

    panels = [
        ("$I$", stokes_mt.I, stokes_ml.I),
        ("$Q/I$", stokes_mt.Q / stokes_mt.I, stokes_ml.Q / stokes_ml.I),
        ("$U/I$", stokes_mt.U / stokes_mt.I, stokes_ml.U / stokes_ml.I),
        ("$V/I$", stokes_mt.V / stokes_mt.I, stokes_ml.V / stokes_ml.I),
    ]

    print(f"B = {magnetic_field_gauss:.0f} G  (S = 0 line; multi-term must equal multi-level):")
    for label, mt_curve, ml_curve in panels:
        print(f"  max|Delta {label}| = {np.max(np.abs(mt_curve - ml_curve)):.2e}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (label, mt_curve, ml_curve) in zip(axes.ravel(), panels):
        ax.plot(reduced_frequency, mt_curve, lw=1.2, color="#1f77b4")
        ax.plot(reduced_frequency, ml_curve, lw=2.8, ls=(0, (1, 1)), color="#d62728")
        ax.set_ylabel(label)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r"reduced frequency $v = (\nu - \nu_0)/\Delta\nu_D$")
    style_key = [
        Line2D([], [], color="#1f77b4", lw=1.2, label="Multi-term atom"),
        Line2D([], [], color="#d62728", lw=2.8, ls=(0, (1, 1)), label="Multi-level atom"),
    ]
    axes[0, 0].legend(handles=style_key, fontsize=8, loc="best")
    fig.suptitle("Multi-term vs multi-level: S=0 line ($^1S_0 \\to {}^1P_1$), $B = 1200$ G")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
    plt.show()
