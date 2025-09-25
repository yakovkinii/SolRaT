import logging
from typing import Union

import numpy as np
from numpy import real
from scipy.linalg import expm
from yatools import logging_config

from src.engine.functions.decorators import log_method
from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE


class ConstantPropertySlab:
    """
    A slab with constant atmospheric properties throughout its depth.
    Solves the radiative transfer equation using DELO method.
    """
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
        delta_v_thermal_cm_sm1: float,
        macroscopic_velocity_cm_sm1: float = 0,
        voigt_a: float = 0,
        initial_stokes: Union[Stokes, float, None] = None,
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

        self.rte = MultiTermAtomRTE(
            level_registry=multi_term_atom_context.level_registry,
            transition_registry=multi_term_atom_context.transition_registry,
            nu=self.nu,
            angles=self.angles,
            magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss,
        )

        self.see.add_all_equations(
            atmosphere_parameters=self.atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(
                chi_B=self.angles.chi_B, theta_B=self.angles.theta_B
            ),
        )

        # Handle initial conditions
        if isinstance(initial_stokes, Stokes):
            assert len(initial_stokes.nu) == len(self.nu), "Initial Stokes vector must have the same frequency grid as the slab"
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = initial_stokes.I
            self.initial_stokes[:, 1, 0] = initial_stokes.Q
            self.initial_stokes[:, 2, 0] = initial_stokes.U
            self.initial_stokes[:, 3, 0] = initial_stokes.V
        elif isinstance(initial_stokes, (int, float)):
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = initial_stokes
        else:
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = 1.0

    @log_method
    def forward(self) -> Stokes:
        """
        Solve radiative transfer through the constant property slab using DELO method.

        Returns:
            Emergent Stokes vector
        """
        stokes = self.initial_stokes

        # Solve statistical equilibrium equations
        rho = self.see.get_solution_direct()

        # Compute radiative transfer coefficients
        rtc = self.rte.compute_all_coefficients(
            atmosphere_parameters=self.atmosphere_parameters,
            rho=rho,
        )

        # DELO method: S=K^-1 * epsilon, expM=expm(K*dtau), new_stokes = S + expM * (stokes - S)
        K_tau = rtc.K_tau()  # [Nν, 4, 4]
        epsilon_tau = rtc.epsilon_tau()[:, :, 0]  # [Nν, 4]

        # Stable solve for source function S at all frequencies
        S = np.stack([
            (np.linalg.solve(K, eps)
             if np.linalg.cond(K) < 1e12
             else (np.linalg.pinv(K) @ eps))
            for K, eps in zip(K_tau, epsilon_tau)
        ])

        # Compute propagation matrix
        expM = np.stack([expm(-K * self.tau) for K in K_tau])  # [Nν,4,4]

        # Apply DELO solution
        stokes = S[:, :, np.newaxis] + np.einsum('nij,njk->nik', expM, stokes - S[:, :, np.newaxis])

        return Stokes(
            nu=self.nu,
            I=real(stokes[:, 0, 0]),
            Q=real(stokes[:, 1, 0]),
            U=real(stokes[:, 2, 0]),
            V=real(stokes[:, 3, 0]),
        )
