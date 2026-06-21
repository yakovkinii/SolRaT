import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter


def build_two_level_model():
    """A J=0 -> J=1 resonance line as a multi-level atom (fast, no Zeeman structure needed)."""
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="1s", J=0, energy_cmm1=0, g=1.0)
    level_registry.register_level(alpha="2p", J=1, energy_cmm1=20_000, g=1.2)

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="2p", J=1),
        level_lower=level_registry.get_level(alpha="1s", J=0),
        einstein_a_ul_sm1=1e7,
    )

    nu_ul = energy_cmm1_to_frequency_sm1(20_000)
    reference_lambda_A_air = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu_ul))

    model = Models.multi_level_atom()
    model = model.configure(
        config=model.Config(
            level_registry=level_registry,
            transition_registry=transition_registry,
            atomic_mass_amu=4.0,
            reference_lambda_A_air=reference_lambda_A_air,
        )
    )
    return model, reference_lambda_A_air, nu_ul


def _run(model, nu, velocity_cm_sm1, theta_v):
    """Uniform, B-free slab with a single uniform bulk velocity (magnitude, direction theta_v)."""
    height_cm = np.linspace(0.0, 200e5, 4)
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_cm,
        temperature_K=8000.0,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=0.0,  # no Zeeman: isolate the Doppler shift in Stokes I
        velocity_cm_sm1=velocity_cm_sm1,
        theta_v=theta_v,  # 0 -> along +z (toward observer); pi -> along -z (away)
        chi_v=0.0,
        delta_v_turbulent_cm_sm1=2.0e5,
        continuum_to_line_ratio=1e-2,
    )
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=10 * np.pi / 180,  # near-vertical: v_los ~ full vertical velocity, observer on +z side
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=2,
        n_phi_quadrature=2,
        max_iterations=5,
        tolerance=1e-3,
    )
    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)
    return atmosphere.forward(initial_stokes=initial_stokes)


def main():
    r"""
    Velocity-sign sanity demo.

    A uniform, magnetic-field-free slab is given a single uniform bulk velocity of 20 km/s,
    once directed along +z (an upflow) and once along -z (a downflow). The observer sits on
    the +z side (near-vertical LOS), so the +z upflow is plasma moving TOWARD the observer.

    Physical expectation (LL04 / standard Doppler): material moving toward the observer is
    BLUESHIFTED, i.e. the line core moves to SHORTER wavelength (Delta lambda_vac < 0). The
    downflow (away) should redshift. The line-core wavelength of each case is printed so the
    direction is unambiguous.

    The projection is fixed in the engine as v_los = -Omega_hat . v (Omega_hat = photon
    propagation / toward-observer direction), which is the form consistent with the bundled
    profile convention (nu = nu_i (1 - v_los/c), positive v_los = redshift). This demo just
    confirms it: the 'upflow (toward)' curve should sit at SHORTER wavelength than static.
    """
    setup_logging()

    model, reference_lambda_A_air, nu_ul = build_two_level_model()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 1.0,
        upper_wavelength_A=reference_lambda_A_air + 1.0,
        step_A=5e-3,
    )
    rest_lambda_vac_A = frequency_sm1_to_lambda_A(nu_ul)

    v = 20.0e5  # 20 km/s
    cases = [
        ("static (v = 0)", 0.0, 0.0),
        ("upflow +z (toward observer)", v, 0.0),
        ("downflow -z (away from observer)", v, np.pi),
    ]

    plotter = StokesPlotter("Velocity-sign check (Stokes I)", reference_lambda_A_air=reference_lambda_A_air)
    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)

    print(f"rest line center: {rest_lambda_vac_A:.4f} A (vac)")
    for label, mag, theta_v in cases:
        emergent = _run(model, nu, mag, theta_v)
        core_nu = nu[int(np.argmin(emergent.I))]
        core_lambda_vac_A = frequency_sm1_to_lambda_A(core_nu)
        d_lambda = core_lambda_vac_A - rest_lambda_vac_A
        print(f"{label:35s} -> line core {core_lambda_vac_A:.4f} A (vac), Delta = {d_lambda:+.4f} A")
        plotter.add_stokes(
            nu=nu,
            stokes=emergent,
            norm=StokesPlotter.Norm.BY_REFERENCE,
            stokes_reference=initial_stokes,
            label=label,
        )

    plotter.show()


if __name__ == "__main__":
    main()
