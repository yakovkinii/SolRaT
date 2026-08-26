import pathlib

import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import get_boundary_stokes_allen
from solrat.atom_model.shared.utility.functions import (
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging

LINE_CENTER_A = 5876.0
HALF_WINDOW_A = 1.0
N_WAVELENGTH = 300

# Thermodynamics shared by every profile: the Hazel runs use one Doppler width and slab temperature,
# so only the field, geometry, height, and optical depth vary between profiles.
TEMPERATURE_K = 10000.0
DELTA_V_TURBULENT_CM_SM1 = 4.7e5
CONTINUUM_DELTA_TAU = 1.0e-8

# Per-profile factor on the optical depth: SolRaT line_delta_tau = TAU_MULTIPLIER * Hazel tau.
# This is because SolRat and Hazel have different tau definitions.
# The Stokes I amplitude therefore is fitted by , but it's just a single point o
# all other points are legitimate comparison.
TAU_MULTIPLIERS = [1, 1, 1]

HAZEL_PROFILES_CSV = pathlib.Path(__file__).with_name("hazel_reference") / "hazel_HeID3_profiles.csv"
HAZEL_D3_WAVELENGTH_A = 5875.9663


def apply_nbar_reduction(radiation_tensor, upper_beta, lower_beta, factor):
    r"""
    Apply Hazel's nbar reduction factor to one transition after Allen tensor construction.
    """
    for transition in radiation_tensor.transition_registry.transitions.values():
        if transition.term_upper.beta == upper_beta and transition.term_lower.beta == lower_beta:
            for K, Q in [(0, 0), (2, 0)]:
                radiation_tensor.set(
                    transition=transition,
                    K=K,
                    Q=Q,
                    value=factor * radiation_tensor.get(transition=transition, K=K, Q=Q),
                )
            return radiation_tensor
    raise ValueError(f"Transition {upper_beta}->{lower_beta} not found.")


def synthesize_solrat(field_gauss, incl_deg, azim_deg, los_deg, height_arcsec, line_delta_tau):
    r"""
    He I D3 Stokes synthesis with SolRaT for one profile, using Hazel's per-transition Allen
    radiation field and incident boundary.

    :return: tuple ``(delta_lambda_A, stokes)`` -- the air-wavelength offset from the line center
        [Angstrom] and the emergent :class:`Stokes`.
    """
    model = PreconfiguredModels.multi_term_atom_HeID3()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=LINE_CENTER_A - HALF_WINDOW_A,
        upper_wavelength_A=LINE_CENTER_A + HALF_WINDOW_A,
        step_A=2.0 * HALF_WINDOW_A / N_WAVELENGTH,
    )
    angles = Angles(
        chi=0.0,
        theta=np.deg2rad(los_deg),
        gamma=0.0,
        chi_B=np.deg2rad(azim_deg),
        theta_B=np.deg2rad(incl_deg),
    )
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=field_gauss,
        temperature_K=TEMPERATURE_K,
        delta_v_turbulent_cm_sm1=DELTA_V_TURBULENT_CM_SM1,
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=height_arcsec)
    radiation_tensor = apply_nbar_reduction(
        radiation_tensor=radiation_tensor,
        upper_beta="3p3",
        lower_beta="2s3",
        factor=0.2,
    )
    slab = ConstantPropertySlabAtmosphere(
        model=model,
        radiation_tensor=radiation_tensor,
        line_delta_tau=line_delta_tau,
        continuum_delta_tau=CONTINUUM_DELTA_TAU,
        angles=angles,
        atmosphere_parameters=atmosphere_parameters,
    )
    emergent = slab.forward(
        initial_stokes=get_boundary_stokes_allen(
            nu_sm1=nu,
            lambda_A=HAZEL_D3_WAVELENGTH_A,
            mu=np.cos(np.deg2rad(los_deg)),
        )
    )
    delta_lambda_a = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu)) - LINE_CENTER_A
    return delta_lambda_a, emergent


def load_hazel_profiles():
    r"""
    Load the multi-profile Hazel reference table (one row per profile and wavelength) with the
    per-profile field, geometry, height, optical depth, and the ``(n, w)`` anisotropy parameters, plus
    the Hazel Stokes spectrum.

    :return: list of per-profile dicts, ordered by the ``profile`` column.
    """
    table = np.genfromtxt(HAZEL_PROFILES_CSV, delimiter=",", names=True)
    profiles = []
    for profile_id in np.unique(table["profile"]):
        rows = table[table["profile"] == profile_id]
        rows = rows[np.argsort(rows["wavelength_A"])]
        meta = rows[0]
        profiles.append(
            {
                "id": int(profile_id),
                "field_gauss": float(meta["field_gauss"]),
                "incl_deg": float(meta["field_incl_deg"]),
                "azim_deg": float(meta["field_azim_deg"]),
                "los_deg": float(meta["los_deg"]),
                "height_arcsec": float(meta["height_arcsec"]),
                "optical_depth": float(meta["optical_depth"]),
                "delta_lambda_A": rows["wavelength_A"] - LINE_CENTER_A,
                "I": rows["I"],
                "Q": rows["Q"],
                "U": rows["U"],
                "V": rows["V"],
            }
        )
    return profiles


def main():
    r"""
    Compare SolRaT and Hazel He I D3 Stokes profiles across several chromospheric-slab configurations
    (different field strengths and geometries), one column per profile, four Stokes rows.

    :return: matplotlib Figure.
    """
    setup_logging()

    profiles = load_hazel_profiles()
    stokes_labels = ["$I / I_{\\mathrm{c}}$", "$Q/I$", "$U/I$", "$V/I$"]

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.8), sharex=True)
    agreement = []
    colors = ["k", "#d62728", "#2ca02c"]
    for index, profile in enumerate(profiles):
        tau_multiplier = TAU_MULTIPLIERS[index] if index < len(TAU_MULTIPLIERS) else 1.0
        delta_lambda_a, solrat = synthesize_solrat(
            profile["field_gauss"],
            profile["incl_deg"],
            profile["azim_deg"],
            profile["los_deg"],
            profile["height_arcsec"],
            profile["optical_depth"] * tau_multiplier,
        )
        color = colors[index % len(colors)]
        label = (
            f"profile {profile['id']}: " f"$|B|$={profile['field_gauss']:.0f} G, LOS={profile['los_deg']:.0f}$^\\circ$"
        )
        solrat_curves = [solrat.I / solrat.I[0], solrat.Q / solrat.I, solrat.U / solrat.I, solrat.V / solrat.I]
        hazel_curves = [
            profile["I"] / profile["I"][0],
            profile["Q"] / profile["I"],
            profile["U"] / profile["I"],
            profile["V"] / profile["I"],
        ]
        for row in range(4):
            axis = axes[row]
            axis.plot(delta_lambda_a, solrat_curves[row], lw=0.9, color=color, label=f"{label} SolRaT")
            axis.plot(
                profile["delta_lambda_A"],
                hazel_curves[row],
                lw=2.4,
                ls=(0, (1, 1)),
                color=color,
                label=f"{label} Hazel",
            )
            axis.axhline(0.0, color="0.7", lw=0.6)
            axis.grid(color="0.88", linewidth=0.5, alpha=0.7)
            axis.set_ylabel(stokes_labels[row])
            axis.set_xlim(-1.0, 0.5)
        for name, solrat_fraction, hazel_fraction in zip(["Q/I", "U/I", "V/I"], solrat_curves[1:], hazel_curves[1:]):
            hazel_on_solrat = np.interp(delta_lambda_a, profile["delta_lambda_A"], hazel_fraction)
            rms = float(np.sqrt(np.mean((solrat_fraction - hazel_on_solrat) ** 2)))
            agreement.append(f"profile {profile['id']} {name}: RMS={rms:.2e}")

    axes[0].legend(fontsize=8, ncol=2, loc="best")
    axes[-1].set_xlabel(r"$\lambda - %.1f$ ($\AA$)" % LINE_CENTER_A)
    fig.align_ylabels(axes)
    fig.tight_layout()
    print("SolRaT vs Hazel (He I D3):")
    for line in agreement:
        print("  " + line)
    return fig


if __name__ == "__main__":
    main()
    plt.show()
