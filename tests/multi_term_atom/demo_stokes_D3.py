import logging
from pathlib import Path

import numpy as np
from numpy import real
from yatools import logging_config

from src.core.engine.functions.looping import FROMTO, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.core.engine.generators.summate import summate
from src.core.physics.functions import lambda_cm_to_frequency_hz
from src.core.physics.rotations import T_K_Q
from src.core.ui.plots.plot_stokes_profiles import StokesPlotter
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
    tau = 1  # normal optical depth at the boundary
    dtau = 0.1  # step size in tau
    # ==================================================================

    # radiation tensor at boundary
    radiation_tensor_prev = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(
        h_arcsec=10
    )

    # Stokes LOS boundary condition at tau_max.
    stokes_los_prev = np.zeros((len(nu), 4, 1), dtype=np.float64)
    stokes_los_prev[:, 0, 0] = radiation_tensor_prev.get_NLTE_n_w_parametrized_stokes_I(
        h_arcsec=10, theta=angles.theta, nu=nu
    )

    def get_closest_index(value, array):
        return (np.abs(array - value)).argmin()

    # Stokes grid bounary condition.
    # we use 2x5 grid, which is exact for electric dipole transitions
    theta_grid = np.acos(np.array([1 / np.sqrt(3), -1 / np.sqrt(3)]))
    chi_grid = np.array([0, 1, 2, 3, 4]) * 2 * np.pi / 5
    stokes_grid_prev = np.zeros((2, 5, len(nu), 4, 1), dtype=np.float64)
    rte_grid = dict()
    angles_grid = dict()
    for i, theta_i in enumerate(theta_grid):
        rte_grid[i] = dict()
        angles_grid[i] = dict()
        for j, chi_j in enumerate(chi_grid):
            stokes_grid_prev[i, j, :, 0, 0] = radiation_tensor_prev.get_NLTE_n_w_parametrized_stokes_I(
                h_arcsec=10, theta=theta_i, nu=nu
            )
            angles_grid[i][j] = Angles(
                chi=chi_j,
                theta=theta_i,
                gamma=0,
                chi_B=angles.chi_B,
                theta_B=angles.theta_B,
            )
            rte_grid[i][j] = MultiTermAtomRTE(
                level_registry=level_registry,
                transition_registry=transition_registry,
                nu=nu,
                angles=angles_grid[i][j],
            )

    # Starting point
    Bz = 1000
    atmosphere_parameters = AtmosphereParameters(
        magnetic_field_gauss=Bz,
        delta_v_thermal_cm_sm1=4_000_00,
    )

    # LOS SEE
    see.add_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=radiation_tensor_prev.rotate_to_magnetic_frame(
            chi_B=angles.chi_B, theta_B=angles.theta_B
        ),
    )
    rho = see.get_solution_direct()

    # LOS RT
    rtc = rte_los.compute_all_coefficients(
        atmosphere_parameters=atmosphere_parameters,
        rho=rho,
    )

    I = np.eye(4, dtype=np.float64)
    I = np.repeat(I[np.newaxis, :, :], len(nu), axis=0)

    Delta = rtc.eta_tau() * dtau / np.cos(angles.theta)
    Kstar_los_prev = rtc.K_tau() - I
    epsilon_los_prev = rtc.epsilon_tau()
    gamma_los_prev = np.exp(-Delta)
    alpha_los_prev = (1 - gamma_los_prev) / Delta - gamma_los_prev
    beta_los_prev = (1 - gamma_los_prev) / Delta

    # GRID SEE
    Kstar_grid_prev = dict()
    epsilon_grid_prev = dict()
    gamma_grid_prev = dict()
    alpha_grid_prev = dict()
    beta_grid_prev = dict()

    w = 2 * np.pi / 5  # Weights for 2x5
    for i, theta_i in enumerate(theta_grid):
        Kstar_grid_prev[i] = dict()
        epsilon_grid_prev[i] = dict()
        gamma_grid_prev[i] = dict()
        alpha_grid_prev[i] = dict()
        beta_grid_prev[i] = dict()
        for j, chi_j in enumerate(chi_grid):
            rtc = rte_grid[i][j].compute_all_coefficients(
                atmosphere_parameters=atmosphere_parameters,
                rho=rho,
            )
            Delta = rtc.eta_tau() * dtau / np.cos(angles.theta)
            Kstar_grid_prev[i][j] = rtc.K_tau() - I
            epsilon_grid_prev[i][j] = rtc.epsilon_tau()
            gamma_grid_prev[i][j] = np.exp(-Delta)
            alpha_grid_prev[i][j] = (1 - gamma_los_prev) / Delta - gamma_los_prev
            beta_grid_prev[i][j] = (1 - gamma_los_prev) / Delta

        plotter.add(
            lambda_A=lambda_A,
            stokes_I=real(stokes_los_prev[:, 0, 0]),
            stokes_Q=real(stokes_los_prev[:, 1, 0]),
            stokes_U=real(stokes_los_prev[:, 2, 0]),
            stokes_V=real(stokes_los_prev[:, 3, 0]),
            reference_lambda_A=reference_lambda_A,
            color="auto",
            label=rf"tau={tau:.2f}",
        )

    # RUN TAU LOOP with DELO
    while tau > 0:
        logging.warning(f"TAU LOOP : {tau=}")
        logging.warning(f"TAU LOOP : {tau=}")
        logging.warning(f"TAU LOOP : {tau=}")
        logging.warning(f"TAU LOOP : {tau=}")
        logging.warning(f"TAU LOOP : {tau=}")
        logging.warning(f"TAU LOOP : {tau=}")
        # Get slab's properties
        Bz = 20000
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz,
            delta_v_thermal_cm_sm1=4_000_00,
        )

        # LOS SEE
        see.add_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_prev,
        )
        rho = see.get_solution_direct()
        rtc = rte_los.compute_all_coefficients(
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
        )
        K = rtc.K_tau()
        epsilon = rtc.epsilon_tau()

        Kstar = K - I
        M = I - beta_los_prev * Kstar
        b = (
            gamma_los_prev * stokes_los_prev
            + alpha_los_prev * (epsilon_los_prev - Kstar_los_prev @ stokes_los_prev)
            + beta_los_prev * epsilon
        )

        # solve Mx=b
        stokes_los_prev = np.linalg.solve(M, b)

        # Update the previous values
        Kstar_los_prev = Kstar
        epsilon_los_prev = epsilon
        Delta = rtc.eta_tau() * dtau / np.cos(angles.theta)
        gamma_los_prev = np.exp(-Delta)
        alpha_los_prev = (1 - gamma_los_prev) / Delta - gamma_los_prev
        beta_los_prev = (1 - gamma_los_prev) / Delta

        # GRID SEE
        for i, theta_i in enumerate(theta_grid):
            for j, chi_j in enumerate(chi_grid):
                rtc = rte_grid[i][j].compute_all_coefficients(
                    atmosphere_parameters=atmosphere_parameters,
                    rho=rho,
                )
                K = rtc.K_tau()
                epsilon = rtc.epsilon_tau()

                Kstar = K - I
                M = I - beta_grid_prev[i][j] * Kstar
                b = (
                    gamma_grid_prev[i][j] * stokes_grid_prev[i, j]
                    + alpha_grid_prev[i][j] * (epsilon_grid_prev[i][j] - Kstar_grid_prev[i][j] @ stokes_grid_prev[i, j])
                    + beta_grid_prev[i][j] * epsilon
                )

                # Update the previous values
                Kstar_grid_prev[i][j] = Kstar
                epsilon_grid_prev[i][j] = epsilon
                Delta = rtc.eta_tau() * dtau / np.cos(angles.theta)
                gamma_grid_prev[i][j] = np.exp(-Delta)
                alpha_grid_prev[i][j] = (1 - gamma_grid_prev[i][j]) / Delta - gamma_grid_prev[i][j]
                beta_grid_prev[i][j] = (1 - gamma_grid_prev[i][j]) / Delta
                stokes_grid_prev[i, j] = np.linalg.solve(M, b)

        radiation_tensor_prev = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(
            h_arcsec=10
        )
        # for selected transition, set the radiation tensor values
        for transition in transition_registry.transitions.values():
            nui = transition.get_mean_transition_frequency_sm1()
            if nui < nu.min() or nui > nu.max():
                continue
            # Get the closest index in nu array
            nu_index = get_closest_index(nui, nu)
            for K, Q in nested_loops(
                K=FROMTO(0, 2),
                Q=PROJECTION("K"),
            ):
                radiation_tensor_prev.set(
                    transition=transition,
                    K=K,
                    Q=Q,
                    value=lambda i, j, stokes_component_index: summate(
                        w
                        * T_K_Q(
                            K=K,
                            Q=Q,
                            stokes_component_index=stokes_component_index,
                            chi=angles_grid[i][j].chi,
                            theta=angles_grid[i][j].theta,
                            gamma=angles_grid[i][j].gamma,
                        )
                        * stokes_grid_prev[i][j][nu_index],
                        i=FROMTO(0, len(theta_grid)),
                        j=FROMTO(0, len(chi_grid)),
                        stokes_component_index=FROMTO(0, 3),
                    ),
                )

        tau -= dtau
        if tau < 0:
            tau = 0

        plotter.add(
            lambda_A=lambda_A,
            stokes_I=real(stokes_los_prev[:, 0, 0]),
            stokes_Q=real(stokes_los_prev[:, 1, 0]),
            stokes_U=real(stokes_los_prev[:, 2, 0]),
            stokes_V=real(stokes_los_prev[:, 3, 0]),
            reference_lambda_A=reference_lambda_A,
            color="auto",
            label=rf"tau={tau:.2f}",
        )

    plotter.show()


if __name__ == "__main__":
    main()
