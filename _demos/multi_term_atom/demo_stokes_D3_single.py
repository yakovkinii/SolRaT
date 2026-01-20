import logging
from pathlib import Path

import numpy as np
from numpy import real
from yatools import logging_config

from src.common.functions import lambda_cm_to_frequency_hz
from src.gui.plots.plot_stokes_profiles import StokesPlotter
from src.multi_term_atom.atomic_data.HeI import fill_precomputed_He_I_D3_data, get_He_I_D3_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


def main():
    """
    This demo shows the calculation of stimulated emission eta_S profiles
    for the He I D3 transition under super-strong magnetic fields.
    This result is closely related to Fig. 8 in Yakovkin & Lozitsky (MNRAS, 2023)
    https://doi.org/10.1093/mnras/stad1816, where these profiles were obtained using HAZEL2.
    In the mentioned paper, the Stokes profiles are shown instead, but they should resemble eta_S for low optical depth.
    """

    logging_config.init(logging.INFO)

    # Load the atomic data for He I D3
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()

    # The calculation itself needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda_A - 1, reference_lambda_A + 1, 5e-4)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

    # Set up the statistical equilibrium equations.
    # Do not pre-compute coefficients, load them from file instead for the purpose of this demo.
    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.as_posix())

    angles = Angles(
        chi=0,
        theta=0,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )
    # Set up the radiative transfer equations
    # Angles input is optional. But since we know the angles in advance,
    # we provide them here to speed up the calculation.
    rte_los = MultiTermAtomRTE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        nu=nu,
        angles=angles,
        # magnetic_field_gauss=...,  # We will vary the magnetic field in this demo, so we cannot provide it here
        # rho=...,  # Rho is dependent on the varying magnetic field, so we cannot provide it here
    )

    # Set up the plotter
    plotter = StokesPlotter(title="He I D3: Stokes profiles")

    # ==================================================================
    # tau = 1  # normal optical depth at the boundary
    # ==================================================================

    # radiation tensor at boundary
    radiation_tensor_prev = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(
        h_arcsec=10
    )

    # Stokes LOS boundary condition at tau_max.
    stokes_los_prev = np.zeros((len(nu), 4, 1), dtype=np.float64)
    stokes_los_prev[:, 0, 0] = 1.0
    # stokes_los_prev[:, 0, 0] = radiation_tensor_prev.get_NLTE_n_w_parametrized_stokes_I(
    #     h_arcsec=10, theta=angles.theta, nu=nu
    # )

    # Starting point
    Bz = 10
    atmosphere_parameters = AtmosphereParameters(
        magnetic_field_gauss=Bz,
        delta_v_thermal_cm_sm1=1_000_00,
    )

    # LOS SEE
    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=radiation_tensor_prev.rotate_to_magnetic_frame(
            chi_B=angles.chi_B, theta_B=angles.theta_B
        ),
    )
    rho = see.get_solution()

    # LOS RT
    rtc = rte_los.compute_all_coefficients(
        atmosphere_parameters=atmosphere_parameters,
        rho=rho,
    )

    eta_for_tau = rtc.get_eta_for_tau()

    stokes_direct = stokes_los_prev.copy()
    stokes_direct_n = stokes_los_prev.copy()
    stokes_tau_direct = stokes_los_prev.copy()

    dtau = -1  # from Sun to the observer, so negative
    dz = -dtau / eta_for_tau  # todo angles

    plotter.add(
        lambda_A=lambda_A,
        stokes_I=real(stokes_los_prev[:, 0, 0]),
        stokes_Q=real(stokes_los_prev[:, 1, 0]),
        stokes_U=real(stokes_los_prev[:, 2, 0]),
        stokes_V=real(stokes_los_prev[:, 3, 0]),
        reference_lambda_A=reference_lambda_A,
        color="black",
        label=rf"initial",
    )

    def direct_stokes_step(current_stokes, K_at_z, epsilon_at_z, dz):
        return current_stokes + (-K_at_z @ current_stokes + epsilon_at_z) * dz

    stokes_direct = direct_stokes_step(current_stokes=stokes_direct, K_at_z=rtc.K(), epsilon_at_z=rtc.compute_epsilon(), dz=dz)

    plotter.add(
        lambda_A=lambda_A,
        stokes_I=real(stokes_direct[:, 0, 0]),
        stokes_Q=real(stokes_direct[:, 1, 0]),
        stokes_U=real(stokes_direct[:, 2, 0]),
        stokes_V=real(stokes_direct[:, 3, 0]),
        reference_lambda_A=reference_lambda_A,
        color="auto",
        label=rf"stokes_direct",
        style="--",
    )

    for _ in range(1000):
        K_at_z = rtc.K()  # This simulates different K at different z
        epsilon_at_z = rtc.compute_epsilon()  # This simulates different epsilon at different z
        stokes_direct_n = direct_stokes_step(
            current_stokes=stokes_direct_n, K_at_z=K_at_z, epsilon_at_z=epsilon_at_z, dz=dz / 1000
        )

    plotter.add(
        lambda_A=lambda_A,
        stokes_I=real(stokes_direct_n[:, 0, 0]),
        stokes_Q=real(stokes_direct_n[:, 1, 0]),
        stokes_U=real(stokes_direct_n[:, 2, 0]),
        stokes_V=real(stokes_direct_n[:, 3, 0]),
        reference_lambda_A=reference_lambda_A,
        color="auto",
        label=rf"stokes_direct_N",
        style=":",
    )

    def tau_direct_stokes_step():
        ...

    stokes_tau_direct = stokes_tau_direct + (rtc.K_tau() @ stokes_tau_direct - rtc.epsilon_tau()) * dtau

    plotter.add(
        lambda_A=lambda_A,
        stokes_I=real(stokes_tau_direct[:, 0, 0]),
        stokes_Q=real(stokes_tau_direct[:, 1, 0]),
        stokes_U=real(stokes_tau_direct[:, 2, 0]),
        stokes_V=real(stokes_tau_direct[:, 3, 0]),
        reference_lambda_A=reference_lambda_A,
        color="purple",
        label=rf"stokes_tau_direct",
        style=".-",
    )

    # ----- ADDED DELO-CONST METHOD -----
    from scipy.linalg import expm

    K_tau = rtc.K_tau()
    epsilon_tau = rtc.epsilon_tau()[:, :, 0]  # Remove last dim

    # Batched solve for S
    S = np.stack([np.linalg.solve(K, eps) if np.linalg.det(K) > 1e-10
                  else np.zeros(4)
                  for K, eps in zip(K_tau, epsilon_tau)])

    # Compute all matrix exponentials
    expM = np.stack([expm(K * dtau) for K in K_tau])

    # Final computation
    stokes_delo = S[:, :, np.newaxis] + np.einsum('ijk,ik->ij', expM,
                                                  stokes_los_prev[:, :, 0] - S)[:, :, np.newaxis]
    stokes_delo = stokes_delo[:, :, np.newaxis]  # Restore last dimension

    plotter.add(
        lambda_A=lambda_A,
        stokes_I=real(stokes_delo[:, 0, 0]),
        stokes_Q=real(stokes_delo[:, 1, 0]),
        stokes_U=real(stokes_delo[:, 2, 0]),
        stokes_V=real(stokes_delo[:, 3, 0]),
        reference_lambda_A=reference_lambda_A,
        color="auto",
        label=rf"DELO-constant",
        style="--",
    )

    # ----- ADDED DELO-LINEAR METHOD -----
    from scipy.linalg import expm, inv

    # Precompute coefficients
    K_tau = rtc.K_tau()  # Shape: (n_nu, 4, 4)
    epsilon_tau = rtc.epsilon_tau()  # Shape: (n_nu, 4, 1)
    stokes_delo_linear = np.zeros_like(stokes_los_prev)  # Initialize output

    for i in range(len(nu)):
        # Extract matrices/vectors for current frequency
        K_tau_i = K_tau[i, :, :]
        epsilon_tau_i = epsilon_tau[i, :, 0]
        I0 = stokes_los_prev[i, :, 0]

        # Compute source function S = K_tau^{-1} @ epsilon_tau
        try:
            S0 = np.linalg.solve(K_tau_i, epsilon_tau_i)
        except np.linalg.LinAlgError:
            S0 = np.zeros(4)

        # Compute M = K_tau * dtau
        M = K_tau_i * dtau

        # Compute matrix exponential
        expM = expm(M)

        # DELO-linear requires additional terms
        # Compute intermediate matrices
        invK = inv(K_tau_i) if not np.isclose(np.linalg.det(K_tau_i), 0) else np.zeros((4, 4))
        P = invK @ (np.eye(4) - expM) - dtau * expM

        # Approximate derivative term (dS/dtau) using current point only
        # This is a simplification since we don't have next point information
        dS_dtau = np.zeros(4)  # Default to zero if no better estimate => DELO const

        # DELO-linear solution
        stokes_delo_linear[i, :, 0] = expM @ I0 + (np.eye(4) - expM) @ S0 + P @ dS_dtau

    plotter.add(
        lambda_A=lambda_A,
        stokes_I=real(stokes_delo_linear[:, 0, 0]),
        stokes_Q=real(stokes_delo_linear[:, 1, 0]),
        stokes_U=real(stokes_delo_linear[:, 2, 0]),
        stokes_V=real(stokes_delo_linear[:, 3, 0]),
        reference_lambda_A=reference_lambda_A,
        color="auto",
        label=rf"DELO-linear",
        style="-.",
    )

    # Delta = rtc.eta_tau() * dtau / np.cos(angles.theta)
    # Kstar_los_prev = rtc.K_tau() - I
    # epsilon_los_prev = rtc.epsilon_tau()
    # gamma_los_prev = np.exp(-Delta)
    # alpha_los_prev = (1 - gamma_los_prev) / Delta - gamma_los_prev
    # beta_los_prev = (1 - gamma_los_prev) / Delta
    #
    #
    #
    # # Get slab's properties
    # # Bz = 20000
    # # atmosphere_parameters = AtmosphereParameters(
    # #     magnetic_field_gauss=Bz,
    # #     delta_v_thermal_cm_sm1=4_000_00,
    # # )
    #
    # # LOS SEE
    # see.add_all_equations(
    #     atmosphere_parameters=atmosphere_parameters,
    #     radiation_tensor_in_magnetic_frame=radiation_tensor_prev.rotate_to_magnetic_frame(
    #         chi_B=angles.chi_B, theta_B=angles.theta_B
    #     ),
    # )
    # rho = see.get_solution_direct()
    # rtc = rte_los.compute_all_coefficients(
    #     atmosphere_parameters=atmosphere_parameters,
    #     rho=rho,
    # )
    # K = rtc.K_tau()
    # epsilon = rtc.epsilon_tau()
    #
    # Kstar = K - I
    # M = I - beta_los_prev * Kstar
    # b = (
    #     gamma_los_prev * stokes_los_prev
    #     + alpha_los_prev * (epsilon_los_prev - Kstar_los_prev @ stokes_los_prev)
    #     + beta_los_prev * epsilon
    # )
    #
    # # solve Mx=b
    # stokes_los_prev = np.linalg.solve(M, b)
    #
    # plotter.add(
    #     lambda_A=lambda_A,
    #     stokes_I=real(stokes_los_prev[:, 0, 0]),
    #     stokes_Q=real(stokes_los_prev[:, 1, 0]),
    #     stokes_U=real(stokes_los_prev[:, 2, 0]),
    #     stokes_V=real(stokes_los_prev[:, 3, 0]),
    #     reference_lambda_A=reference_lambda_A,
    #     color="auto",
    #     label=rf"tau=1step",
    # )

    plotter.show()


if __name__ == "__main__":
    main()
