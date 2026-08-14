import pathlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging

# He I D3 (5876 A) single-slab synthesis compared against Hazel. Parameters mirror
# C:\ubuntu\hazel\solrat_validation\main.py -- keep the two in sync. Inclined line of sight and a
# strong, inclined field so all four Stokes are non-trivial: at |B| = 8000 G the Zeeman splitting is
# comparable to the Doppler width, so the transverse-Zeeman Q and U grow to be comparable to V and
# dominate over the (field-independent, Hanle-saturated) scattering polarization.
LINE_CENTER_A = 5876.0
HALF_WINDOW_A = 1.0
N_WAVELENGTH = 300
MAGNETIC_FIELD_GAUSS = 8000.0
FIELD_INCLINATION_DEG = 80.0        # theta_B from the vertical
FIELD_AZIMUTH_DEG = 45.0            # chi_B
LOS_INCLINATION_DEG = 60.0          # theta of the line of sight from the vertical
OPTICAL_DEPTH = 1.02               # tuned so the SolRaT line depth matches Hazel (does not have to equal Hazel tau)
DOPPLER_VELOCITY_KM_S = 8.0
HEIGHT_ARCSEC = 10.0
TEMPERATURE_K = 10000.0
DELTA_V_TURBULENT_CM_SM1 = 4.7e5    # with the temperature, gives ~8 km/s He Doppler width (Hazel deltav)
CONTINUUM_TEMPERATURE_K = 6000.0

HAZEL_REFERENCE_CSV = pathlib.Path(__file__).with_name("hazel_reference") / "hazel_HeID3.csv"


def synthesize_solrat():
    r"""
    He I D3 Stokes synthesis with SolRaT, matching the Hazel single-slab setup.

    :return: tuple ``(delta_lambda_A, stokes)`` -- the wavelength offset from the line center
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
        theta=np.deg2rad(LOS_INCLINATION_DEG),
        gamma=0.0,
        chi_B=np.deg2rad(FIELD_AZIMUTH_DEG),
        theta_B=np.deg2rad(FIELD_INCLINATION_DEG),
    )
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=MAGNETIC_FIELD_GAUSS,
        temperature_K=TEMPERATURE_K,
        delta_v_turbulent_cm_sm1=DELTA_V_TURBULENT_CM_SM1,
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(
        h_arcsec=HEIGHT_ARCSEC
    )
    atmosphere = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            line_delta_tau=OPTICAL_DEPTH,
            continuum_delta_tau=1.0e-3,
            angles=angles,
            atmosphere_parameters=atmosphere_parameters,
        )
    )
    emergent = atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=CONTINUUM_TEMPERATURE_K))
    delta_lambda_a = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu)) - LINE_CENTER_A
    return delta_lambda_a, emergent


def load_hazel_reference():
    r"""
    Load the Hazel reference spectrum copied into ``hazel_reference/hazel_HeID3.csv``, if present.

    :return: dict with ``delta_lambda_A``, ``I``, ``Q``, ``U``, ``V`` arrays, or ``None`` if absent.
    """
    table = np.genfromtxt(HAZEL_REFERENCE_CSV, delimiter=",", names=True)
    return {
        "delta_lambda_A": table["wavelength_A"] - LINE_CENTER_A,
        "I": table["I"], "Q": table["Q"], "U": table["U"], "V": table["V"],
    }  # fmt: skip


def main():
    r"""
    Compare SolRaT and Hazel He I D3 Stokes profiles for a strong, inclined field.

    Synthesizes the line with SolRaT for the same single-slab parameters as the Hazel script
    (``C:\ubuntu\hazel\solrat_validation\main.py``) and overlays the Hazel reference copied into
    ``hazel_reference/hazel_HeID3.csv``. Stokes I is continuum-normalized and Q, U, V shown as
    fractions of I; the per-Stokes relative error (RMS difference over SolRaT peak) is printed. At
    |B| = 8000 G all four Stokes are non-trivial and the linear polarization is regular transverse
    Zeeman. The residual (largest in V) reflects the multi-term LS treatment of the fine-structure line
    strengths (one term A_ul distributed by LS 6j, versus Hazel's per-J A values) and incomplete
    Paschen-Back; Q and U are tied to the positive-Q reference direction.

    :return: the matplotlib Figure with the four Stokes panels (not shown; the caller decides whether
        to display or save it).
    """
    setup_logging()

    delta_lambda_a, solrat = synthesize_solrat()
    solrat_panels = [
        ("$I / I_{\\mathrm{c}}$", solrat.I / solrat.I[0]),
        ("$Q/I$", solrat.Q / solrat.I),
        ("$U/I$", solrat.U / solrat.I),
        ("$V/I$", solrat.V / solrat.I),
    ]
    hazel = load_hazel_reference()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, (label, curve) in zip(axes.ravel(), solrat_panels):
        axis.plot(delta_lambda_a, curve, lw=1.4, color="#1f77b4")
        axis.set_ylabel(label)
        axis.axhline(0.0, color="0.7", lw=0.6)
        axis.grid(alpha=0.3)

    hazel_panels = [
        hazel["I"] / hazel["I"][0],
        hazel["Q"] / hazel["I"],
        hazel["U"] / hazel["I"],
        hazel["V"] / hazel["I"],
    ]
    for axis, curve in zip(axes.ravel(), hazel_panels):
        axis.plot(hazel["delta_lambda_A"], curve, lw=2.6, ls=(0, (1, 1)), color="k")
    solrat_fractions = [panel[1] for panel in solrat_panels[1:]]  # Q/I, U/I, V/I
    for label, solrat_fraction, hazel_fraction in zip(["Q/I", "U/I", "V/I"], solrat_fractions, hazel_panels[1:]):
        hazel_on_solrat = np.interp(delta_lambda_a, hazel["delta_lambda_A"], hazel_fraction)
        peak = float(np.max(np.abs(solrat_fraction)))
        rms = float(np.sqrt(np.mean((solrat_fraction - hazel_on_solrat) ** 2)))
        print(f"  {label}: SolRaT peak={peak:.2e}  RMS(SolRaT-Hazel)={rms:.2e}  relative={rms / peak:.1%}")

    for axis in axes[1]:
        axis.set_xlabel(r"$\lambda - %.1f$ ($\AA$)" % LINE_CENTER_A)
    style_key = [Line2D([], [], color="#1f77b4", lw=1.4, label="SolRaT")]
    style_key.append(Line2D([], [], color="k", lw=2.6, ls=(0, (1, 1)), label="Hazel"))
    axes[0, 0].legend(handles=style_key, fontsize=9, loc="best")
    fig.suptitle(f"He I D3, $|B|={MAGNETIC_FIELD_GAUSS:.0f}$ G inclined: SolRaT vs Hazel")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
    plt.show()
