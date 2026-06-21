import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter


def main():
    """
    Self-consistent NLTE synthesis of the He I D3 multiplet through a height-stratified
    atmosphere.

    This is the same formalism-agnostic engine as the multi-level resonance-line demo, here
    driven with the multi-term He I D3 model: temperature, He triplet number density, the
    magnetic-field vector, and a vertical macroscopic-velocity gradient all vary with
    geometric height, and the scattering radiation tensor J^K_Q is solved self-consistently
    (rather than imposed via the {n, w} parametrization used by the constant-property slab).

    The D3 multiplet RTE is heavy, so the grid is kept small (few depths/rays, coarse nu).
    Tune number_density_cm3 so the printed observer optical depth lands where you want it;
    line opacity scales linearly with N(z). Expect this to take noticeably longer than the
    two-level demo.
    """
    setup_logging()

    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.6,
        upper_wavelength_A=reference_lambda_A_air + 1.0,
        step_A=5e-3,  # coarse: the multi-term D3 RTE is expensive per call
    )

    # Geometric height grid [cm]: z[0] = lower boundary (photosphere side), z[-1] = observer.
    n_depth = 8
    height_cm = np.linspace(0.0, 1500e5, n_depth)  # 0 .. 1500 km chromospheric slab

    # Continuously varying atmosphere. Profiles may be scalars, arrays, or callables f(z_cm).
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=height_cm,
        temperature_K=lambda z: 8000.0 + 4000.0 * (z / height_cm[-1]),  # 8000 -> 12000 K
        number_density_cm3=lambda z: 5.0e9 * np.exp(-z / 800e5),  # He triplet lower level, falls off
        magnetic_field_gauss=100.0,
        theta_B=60 * np.pi / 180,  # inclined field -> linear + circular polarization
        chi_B=20 * np.pi / 180,
        velocity_cm_sm1=lambda z: 1.0e5 * (z / height_cm[-1]),  # 0 -> 1 km/s upflow
        theta_v=0.0,  # vertical
        chi_v=0.0,
        delta_v_turbulent_cm_sm1=5.0e5,  # He lines are broad
        voigt_a=0.0,
        continuum_to_line_ratio=1e-2,
    )

    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=50 * np.pi / 180,
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=2,
        n_phi_quadrature=2,
        max_iterations=8,
        tolerance=1e-3,
    )

    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)
    emergent = atmosphere.forward(initial_stokes=initial_stokes)

    print(f"NLTE iterations used : {atmosphere.iterations_used}")
    print(f"NLTE final residual  : {atmosphere.final_residual:.3e}")
    print(f"observer optical depth: {atmosphere.tau_grid[-1]:.3e}")

    plotter = StokesPlotter(
        "Stratified NLTE He I D3 (T, B, v(z) gradients)", reference_lambda_A_air=reference_lambda_A_air
    )
    plotter.add_stokes(
        nu=nu,
        stokes=emergent,
        norm=StokesPlotter.Norm.BY_REFERENCE,
        stokes_reference=initial_stokes,
        label="NLTE stratified He I D3",
    )
    plotter.show()


if __name__ == "__main__":
    main()
