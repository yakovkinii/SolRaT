import numpy as np
from numpy import real

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range, get_planck_BP
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter


def main():
    """
    This demo shows how the DELO solver works against the different more primitive finite difference method.
    """
    setup_logging()

    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 1,
        upper_wavelength_A=reference_lambda_A_air + 1,
        step_A=1e-3,
    )

    angles = Angles(
        chi=30 * np.pi / 180,
        theta=40 * np.pi / 180,
        gamma=50 * np.pi / 180,
        chi_B=10 * np.pi / 180,
        theta_B=20 * np.pi / 180,
    )

    plotter = StokesPlotter(
        "Comparison of DELO and Finite Difference integration", reference_lambda_A_air=reference_lambda_A_air
    )

    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=10000,
        temperature_K=6000,
    )

    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(
        h_arcsec=30,
    )

    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)

    line_delta_tau = 0.2
    continuum_delta_tau = 0.001
    atmosphere = ConstantPropertySlabAtmosphere(
        model=model,
        radiation_tensor=radiation_tensor,
        line_delta_tau=line_delta_tau,
        continuum_delta_tau=continuum_delta_tau,
        angles=angles,
        atmosphere_parameters=atmosphere_parameters,
    )

    delo_stokes = atmosphere.forward(initial_stokes=initial_stokes)
    plotter.add_stokes(
        nu=nu,
        stokes=delo_stokes,
        norm=StokesPlotter.Norm.BY_REFERENCE,
        stokes_reference=initial_stokes,
        label="DELO",
    )

    # Reuse the same RTC using different methods and reconstruct the K and epsilon
    rtc = atmosphere.rtc

    #  dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

    K_tau_line = rtc.K_tau()  # [Nν, 4, 4]
    epsilon_tau_line = rtc.epsilon_tau()[:, :, 0]  # [Nν, 4]

    eta_LC = line_delta_tau / continuum_delta_tau

    # Add continuum
    K_tau_line[:, 0, 0] += 1 / eta_LC
    K_tau_line[:, 1, 1] += 1 / eta_LC
    K_tau_line[:, 2, 2] += 1 / eta_LC
    K_tau_line[:, 3, 3] += 1 / eta_LC
    epsilon_tau_line[:, 0] += get_planck_BP(nu_sm1=nu, temperature_K=atmosphere_parameters.temperature_K) / eta_LC

    # Rename to be explicit
    K_tau = K_tau_line
    epsilon_tau = epsilon_tau_line

    # Construct initial conditions:
    stokes = np.zeros((len(nu), 4, 1), dtype=np.float64)
    stokes[:, 0, 0] = initial_stokes.I
    stokes[:, 1, 0] = initial_stokes.Q
    stokes[:, 2, 0] = initial_stokes.U
    stokes[:, 3, 0] = initial_stokes.V

    # Solve the transfer equation
    # dStokes/dtau_line = K_tau * Stokes - epsilon_tau

    def direct_stokes_step(current_stokes, K, epsilon, dtau):
        return current_stokes + (K @ current_stokes - epsilon[:, :, None]) * dtau

    n_steps = 20
    dtau = -line_delta_tau / n_steps
    for i in range(n_steps):
        stokes = direct_stokes_step(current_stokes=stokes, K=K_tau, epsilon=epsilon_tau, dtau=dtau)

        if i % 2 == 1:
            plotter.add_stokes(
                nu=nu,
                stokes=Stokes(
                    nu=nu,
                    I=real(stokes[:, 0, 0]),
                    Q=real(stokes[:, 1, 0]),
                    U=real(stokes[:, 2, 0]),
                    V=real(stokes[:, 3, 0]),
                ),
                norm=StokesPlotter.Norm.BY_REFERENCE,
                stokes_reference=initial_stokes,
                label=f"FD (step #{i+1}/{n_steps})",
                linewidth=0.5,
            )

    fd_final = np.real(stokes[:, :, 0])  # [Nnu, 4]: I, Q, U, V after the finite-difference steps
    delo_array = np.stack([delo_stokes.I, delo_stokes.Q, delo_stokes.U, delo_stokes.V], axis=1)
    print(
        f"DELO vs {n_steps}-step finite difference: max|FD - DELO| / max|DELO| = "
        f"{float(np.max(np.abs(fd_final - delo_array)) / np.max(np.abs(delo_array))):.2e}"
    )
    plotter.show()


if __name__ == "__main__":
    main()
