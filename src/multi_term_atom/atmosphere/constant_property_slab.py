import logging
from pathlib import Path

import numpy as np
from numpy import real
from scipy.linalg import expm
from yatools import logging_config

from src.engine.functions.decorators import log_method
from src.common.functions import lambda_cm_to_frequency_hz
from src.gui.plots.plot_stokes_profiles import StokesPlotter
from src.multi_term_atom.atomic_data.HeI import get_He_I_D3_data, fill_precomputed_He_I_D3_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class Stokes:
    def __init__(self, nu: np.ndarray,  I: np.ndarray, Q: np.ndarray, U: np.ndarray, V: np.ndarray):
        self.nu = nu
        self.I = I
        self.Q = Q
        self.U = U
        self.V = V


class MultiTermAtomContext:
    def __init__(self,
                 level_registry: LevelRegistry, transition_registry: TransitionRegistry,
                 statistical_equilibrium_equations: MultiTermAtomSEE,
                    lambda_A: np.ndarray,
                    reference_lambda_A: float,):
        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.statistical_equilibrium_equations = statistical_equilibrium_equations
        self.lambda_A = lambda_A
        self.reference_lambda_A = reference_lambda_A

class ConstantPropertySlab:
    def __init__(
        self,
        multi_term_atom_context: MultiTermAtomContext,
        radiation_tensor: RadiationTensor,
        tau: float,
        chi: float,
        theta: float,
        gamma: float,
        magnetic_field_gauss: float,
        chi_B: float,
        theta_B: float,
        delta_v_thermal_cm_sm1:float,
        macroscopic_velocity_cm_sm1:float=0,
        voigt_a:float=0,
        dtau: float = -0.1,
    ):
        self.see = multi_term_atom_context.statistical_equilibrium_equations
        self.lambda_A = multi_term_atom_context.lambda_A
        self.reference_lambda_A = multi_term_atom_context.reference_lambda_A
        self.nu = lambda_cm_to_frequency_hz(self.lambda_A * 1e-8)

        self.atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=magnetic_field_gauss,
            delta_v_thermal_cm_sm1=delta_v_thermal_cm_sm1,
            macroscopic_velocity_cm_sm1=macroscopic_velocity_cm_sm1,
            voigt_a=voigt_a,
        )
        self.radiation_tensor = radiation_tensor
        self.angles = Angles(
            chi=chi,
            theta=theta,
            gamma=gamma,
            chi_B=chi_B,
            theta_B=theta_B,
        )
        self.tau = tau
        self.dtau = dtau
        self.rte = MultiTermAtomRTE(
            level_registry=multi_term_atom_context.level_registry,
            transition_registry=multi_term_atom_context.transition_registry,
            nu=self.nu,
            angles=self.angles,
            magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss,
        )

    @log_method
    def forward(self)->Stokes:

        stokes_los_prev = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
        stokes_los_prev[:, 0, 0] = 1.0

        # LOS SEE
        self.see.add_all_equations(
            atmosphere_parameters=self.atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(
                chi_B=self.angles.chi_B, theta_B=self.angles.theta_B
            ),
        )
        rho = self.see.get_solution_direct()

        # LOS RT
        rtc = self.rte.compute_all_coefficients(
            atmosphere_parameters=self.atmosphere_parameters,
            rho=rho,
        )

        # ----- DELO-CONST METHOD -----

        K_tau = rtc.K_tau()
        epsilon_tau = rtc.epsilon_tau()[:, :, 0]  # Remove last dim

        # Batched solve for S
        S = np.stack(
            [np.linalg.solve(K, eps) if np.linalg.det(K) > 1e-10 else np.zeros(4) for K, eps in zip(K_tau, epsilon_tau)]
        )

        # Compute all matrix exponentials
        arrays = [expm(K * self.dtau) for K in K_tau]
        expM = np.stack(arrays)

        # Final computation
        stokes_delo = (
            S[:, :, np.newaxis] + np.einsum("ijk,ik->ij", expM, stokes_los_prev[:, :, 0] - S)[:, :, np.newaxis]
        )
        stokes_delo = stokes_delo[:, :, np.newaxis]  # Restore last dimension

        return Stokes(
            nu=self.nu,
            I=real(stokes_delo[:, 0, 0]),
            Q=real(stokes_delo[:, 1, 0]),
            U=real(stokes_delo[:, 2, 0]),
            V=real(stokes_delo[:, 3, 0]),
        )





if __name__ == '__main__':



        # plotter = StokesPlotter(title="He I D3: Stokes profiles")
        #
        #
        # logging_config.init(logging.INFO)
        #
        # level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()
        #
        # lambda_A = np.arange(reference_lambda_A - 1, reference_lambda_A + 1, 5e-4)
        # nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)
        #
        # # Set up the statistical equilibrium equations.
        # # Do not pre-compute coefficients, load them from file instead for the purpose of this demo.
        # see = MultiTermAtomSEE(
        #     level_registry=level_registry,
        #     transition_registry=transition_registry,
        #     precompute=False,
        # )
        # fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.parent.as_posix())
        #
        # angles = Angles(
        #     chi=0,
        #     theta=0,
        #     gamma=0,
        #     chi_B=0,
        #     theta_B=0,
        # )
        # # Set up the radiative transfer equations
        # # Angles input is optional. But since we know the angles in advance,
        # # we provide them here to speed up the calculation.
        # rte_los = MultiTermAtomRTE(
        #     level_registry=level_registry,
        #     transition_registry=transition_registry,
        #     nu=nu,
        #     angles=angles,
        #     # magnetic_field_gauss=...,  # We will vary the magnetic field in this demo, so we cannot provide it here
        #     # rho=...,  # Rho is dependent on the varying magnetic field, so we cannot provide it here
        # )
        #
        # # Set up the plotter
        # # plotter = StokesPlotter(title="He I D3: Stokes profiles")
        #
        # slab = ConstantPropertySlab(
        #     nu=nu,
        #     lambda_A=lambda_A,
        #     see=see,
        #     rte=rte_los,
        #     atmosphere_parameters=AtmosphereParameters(
        #         magnetic_field_gauss=10,  # Placeholder, will be overridden in the loop below
        #         delta_v_thermal_cm_sm1=5_000_00,
        #     ),
        #     radiation_tensor=RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(
        #         h_arcsec=0.725 * 10_000  # Height in arcsec
        #     ),
        #     angles=angles,
        #     reference_lambda_A=reference_lambda_A,
        #     tau=1.0,
        #     dtau=-1,
        # )
        # stokes = slab.forward()
        #
        #
        # plotter.add(
        #     lambda_A=self.lambda_A,
        #     stokes_I=real(stokes_delo[:, 0, 0]),
        #     stokes_Q=real(stokes_delo[:, 1, 0]),
        #     stokes_U=real(stokes_delo[:, 2, 0]),
        #     stokes_V=real(stokes_delo[:, 3, 0]),
        #     reference_lambda_A=self.reference_lambda_A,
        #     color="auto",
        #     label=rf"DELO-constant",
        #     style="--",
        # )
        #
        # plotter.show()