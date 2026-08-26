import numpy as np
from matplotlib import pyplot as plt

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

UPPER_ENERGY_CMM1 = 20_000.0
EINSTEIN_A_UL_SM1 = 1.0e7
ATOMIC_MASS_AMU = 56.0
TEMPERATURE_K = 6000.0
DELTA_V_TURBULENT_CM_SM1 = 2.0e5
VOIGT_A = 0.05


def reference_frequency_and_wavelength():
    r"""
    Transition frequency and air reference wavelength of the shared line.
    """
    nu0 = energy_cmm1_to_frequency_sm1(UPPER_ENERGY_CMM1)
    return nu0, lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu0))


def build_multi_term_s0(reference_lambda_A_air: float, lte: bool, j_constrained: bool):
    r"""
    Build :math:`^1S_0 \to {}^1P_1` as a multi-term atom.

    The transition registers a redundant :math:`J` constraint. For :math:`S=0` each term has only one
    :math:`J`, so the constrained and all-branches multi-term atoms should be identical.
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
        lower_J_constraint=[0],
        upper_J_constraint=[1],
    )
    config = MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=reference_lambda_A_air,
        atomic_mass_amu=ATOMIC_MASS_AMU,
        j_constrained=j_constrained,
    )
    model = Models.multi_term_atom_lte() if lte else Models.multi_term_atom()
    return model.configure(config=config)


def build_multi_level_s0(reference_lambda_A_air: float, lte: bool):
    r"""
    Build the same :math:`J=0 \to 1` line as a multi-level atom.
    """
    level_registry = MultiLevelLevelRegistry()
    level_registry.register_level(alpha="lower", J=0, energy_cmm1=0.0, g=0.0)
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
    model = Models.multi_level_atom_lte() if lte else Models.multi_level_atom()
    return model.configure(config=config)


def synthesize(model, nu: np.ndarray, angles: Angles, magnetic_field_gauss: float) -> Stokes:
    r"""
    Prescribed-radiation constant-property slab synthesis.
    """
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=magnetic_field_gauss,
        temperature_K=TEMPERATURE_K,
        delta_v_turbulent_cm_sm1=DELTA_V_TURBULENT_CM_SM1,
        voigt_a=VOIGT_A,
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config)
    if hasattr(radiation_tensor, "fill_NLTE_n_w_allen"):
        radiation_tensor = radiation_tensor.fill_NLTE_n_w_allen(h_arcsec=30)
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
    Six-curve :math:`S=0` diagnostic matching the quartet LTE/NLTE comparison.

    The three non-LTE curves should coincide, and the two LTE curves should coincide, because the
    multi-term atom has one :math:`J` per term and therefore reduces to the multi-level atom.
    """
    setup_logging()

    magnetic_field_gauss = 50.0
    angles = Angles(chi=0.0, theta=np.pi / 3, gamma=0.0, chi_B=0.0, theta_B=0.0)
    nu0, reference_lambda_A_air = reference_frequency_and_wavelength()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.35,
        upper_wavelength_A=reference_lambda_A_air + 0.35,
        step_A=1e-3,
    )

    reference_model = build_multi_term_s0(reference_lambda_A_air, lte=False, j_constrained=True)
    delta_v_thermal_cm_sm1 = reference_model.AtmosphereParameters(
        model_config=reference_model.config,
        magnetic_field_gauss=magnetic_field_gauss,
        temperature_K=TEMPERATURE_K,
        delta_v_turbulent_cm_sm1=DELTA_V_TURBULENT_CM_SM1,
        voigt_a=VOIGT_A,
    ).delta_v_thermal_cm_sm1
    reduced_frequency = (nu - nu0) / (nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1)

    curves = (
        (
            "Multi-term, $J$-constrained (non-LTE)",
            build_multi_term_s0(reference_lambda_A_air, lte=False, j_constrained=True),
            "k",
            "-",
        ),
        (
            "Multi-term, all branches (non-LTE)",
            build_multi_term_s0(reference_lambda_A_air, lte=False, j_constrained=False),
            "0.45",
            (0, (4, 1)),
        ),
        ("Multi-level (non-LTE)", build_multi_level_s0(reference_lambda_A_air, lte=False), "#d62728", "--"),
        (
            "Multi-term, $J$-constrained (LTE)",
            build_multi_term_s0(reference_lambda_A_air, lte=True, j_constrained=True),
            "g",
            (0, (2, 3)),
        ),
        (
            "Multi-term, all branches (LTE)",
            build_multi_term_s0(reference_lambda_A_air, lte=True, j_constrained=False),
            "0.45",
            (0, (3, 2)),
        ),
        ("Multi-level (LTE)", build_multi_level_s0(reference_lambda_A_air, lte=True), "#2ca02c", (0, (1, 1))),
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    qi_by_label = {}
    for label, model, color, linestyle in curves:
        stokes = synthesize(model, nu, angles, magnetic_field_gauss)
        qi = 100.0 * stokes.Q / stokes.I
        qi_by_label[label] = qi
        linewidth = 2.2 if label.endswith("(LTE)") else 1.6
        ax.plot(reduced_frequency, qi, lw=linewidth, color=color, ls=linestyle, label=label)

    nlte_reference = qi_by_label["Multi-level (non-LTE)"]
    lte_reference = qi_by_label["Multi-level (LTE)"]
    nlte_rms = [
        float(np.sqrt(np.mean((qi_by_label[label] - nlte_reference) ** 2)))
        for label in (
            "Multi-term, $J$-constrained (non-LTE)",
            "Multi-term, all branches (non-LTE)",
        )
    ]
    lte_rms = [
        float(np.sqrt(np.mean((qi_by_label[label] - lte_reference) ** 2)))
        for label in (
            "Multi-term, $J$-constrained (LTE)",
            "Multi-term, all branches (LTE)",
        )
    ]

    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"$(\nu - \nu_0)/\Delta\nu_D$")
    ax.set_ylabel("$100\\,Q/I$")
    ax.grid(color="0.88", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    print(
        "S=0 LTE/NLTE six-curve check: "
        f"RMS MT constrained NLTE - ML NLTE = {nlte_rms[0]:.2e}, "
        f"RMS MT all-branches NLTE - ML NLTE = {nlte_rms[1]:.2e}, "
        f"RMS MT constrained LTE - ML LTE = {lte_rms[0]:.2e}, "
        f"RMS MT all-branches LTE - ML LTE = {lte_rms[1]:.2e}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
