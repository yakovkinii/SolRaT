import matplotlib.pyplot as plt
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
    r"""
    A J=0 -> J=1 resonance line as a multi-level atom.
    """
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
    return model, reference_lambda_A_air


def main():
    r"""
    Demo: self-consistent NLTE synthesis through a height-stratified atmosphere with
    continuously varying temperature, magnetic field, and a vertical macroscopic-velocity
    gradient (a bulk upflow that increases with height).

    Every physical parameter varies with geometric height; opacity is set by the absorber
    number-density profile N(z). The velocity is a full vector field (here vertical), so each
    quadrature ray sees its own line-of-sight projection of the velocity, and the absorption
    profile is Doppler-shifted accordingly per ray and per depth.
    """
    setup_logging()

    model, reference_lambda_A_air = build_two_level_model()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 1.0,
        upper_wavelength_A=reference_lambda_A_air + 1.0,
        step_A=2e-3,
    )

    # Geometric height grid [cm]: z[0] = lower boundary (photosphere), z[-1] = observer side.
    n_depth = 20
    height_cm = np.linspace(0.0, 300e5, n_depth)  # 0 .. 300 km

    # Continuously varying atmosphere. Profiles may be scalars, arrays, or callables f(z_cm).
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_cm,
        temperature_K=lambda z: 5000.0 + 3000.0 * (z / height_cm[-1]),  # 5000 -> 8000 K
        number_density_cm3=lambda z: 1.0e11 * np.exp(-z / 150e5),  # falls off with height
        magnetic_field_gauss=300.0,
        theta_B=20 * np.pi / 180,
        chi_B=10 * np.pi / 180,
        velocity_cm_sm1=lambda z: 2.0e5 * (z / height_cm[-1]),  # 0 -> 2 km/s upflow
        theta_v=0.0,  # vertical (along +z)
        chi_v=0.0,
        delta_v_turbulent_cm_sm1=2.0e5,
        voigt_a=0.0,
        continuum_to_line_ratio=1e-2,
    )

    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=40 * np.pi / 180,
        los_chi=30 * np.pi / 180,
        los_gamma=0.0,
        n_mu_quadrature=4,
        n_phi_quadrature=3,
        max_iterations=15,
        tolerance=1e-3,
    )

    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)
    emergent = atmosphere.forward(initial_stokes=initial_stokes)

    print(f"NLTE iterations used : {atmosphere.iterations_used}")
    print(f"NLTE final residual  : {atmosphere.final_residual:.3e}")
    print(f"vertical optical thickness: {atmosphere.tau_grid[-1]:.3e}")

    # Physical stratification against the vertical (line-center) optical depth.
    tau = atmosphere.tau_grid
    _, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes[0, 0].plot(tau, stratification.temperature_K, marker=".")
    axes[0, 0].set_ylabel("temperature [K]")
    axes[0, 1].plot(tau, stratification.number_density_cm3, marker=".")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"number density [cm$^{-3}$]")
    axes[1, 0].plot(tau, stratification.magnetic_field_gauss, marker=".")
    axes[1, 0].set_ylabel("magnetic field [G]")
    axes[1, 1].plot(tau, stratification.velocity_cm_sm1 / 1e5, marker=".")
    axes[1, 1].set_ylabel("macroscopic velocity [km/s]")
    for ax in axes[1]:
        ax.set_xlabel(r"vertical optical depth  $\tau$")
    plt.suptitle("Atmosphere stratification vs optical depth")
    plt.tight_layout()

    plotter = StokesPlotter(
        "Stratified NLTE atmosphere (velocity gradient)", reference_lambda_A_air=reference_lambda_A_air
    )
    plotter.add_stokes(
        nu=nu,
        stokes=emergent,
        norm=StokesPlotter.Norm.BY_REFERENCE,
        stokes_reference=initial_stokes,
        label="NLTE stratified (T, B, v(z) gradients)",
    )
    plotter.show()


if __name__ == "__main__":
    main()
