import logging
from typing import Union

import numpy as np
from numpy import real
from scipy.linalg import expm
from yatools import logging_config

from src.common.functions import get_planck_BP_vector, lambda_cm_to_frequency_hz
from src.engine.functions.decorators import log_method
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import (
    MultiTermAtomSEE,
    MultiTermAtomSEELTE,
)


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
        j_constrained=False,
        continuum_opacity: Union[float, np.ndarray] = 0.0,  # [cm⁻¹] at each frequency
        temperature_K=None,
        continuum_source_function: Union[float, np.ndarray] = None,  # B_ν(T) [erg/cm²/s/sr/Hz]
    ):
        # Todo copy()
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
            precompute=False,
            j_constrained=j_constrained,
        )

        if isinstance(self.see, MultiTermAtomSEE):
            self.see.fill_all_equations(
                atmosphere_parameters=self.atmosphere_parameters,
                radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(
                    chi_B=self.angles.chi_B, theta_B=self.angles.theta_B
                ),
            )
        else:
            assert isinstance(self.see, MultiTermAtomSEELTE)

        # Handle initial conditions
        if isinstance(initial_stokes, Stokes):
            assert len(initial_stokes.nu) == len(
                self.nu
            ), "Initial Stokes vector must have the same frequency grid as the slab"
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

        n_nu = len(self.nu)
        if np.isscalar(continuum_opacity):
            self.continuum_opacity = np.full(n_nu, continuum_opacity)
        else:
            self.continuum_opacity = continuum_opacity

        if continuum_source_function is None:
            if temperature_K is None:
                if np.any(continuum_opacity != 0):
                    raise ValueError("Need temperature if continuum opacity is non-zero")
                self.continuum_source_function = np.full(n_nu, 0)
            else:
                self.continuum_source_function = get_planck_BP_vector(nu_sm1=self.nu, T_K=temperature_K)
        elif np.isscalar(continuum_source_function):
            self.continuum_source_function = np.full(n_nu, continuum_source_function)
        else:
            self.continuum_source_function = continuum_source_function

    @log_method
    def forward(self, initial_stokes: Stokes = None) -> Stokes:
        """
        Solve radiative transfer through the constant property slab using DELO method.

        Returns:
            Emergent Stokes vector



                RT matrix K related to the tau-propagation equation

        .. math::
            dStokes/dz = - K * Stokes + epsilon - K_continuum Stokes + epsilon_continuum

            dz = - dtau_line / eta_line

        where _line means line-center frequency.

        .. math::
            dStokes/dtau_line = - K_tau_line * Stokes + epsilon_tau_line - K_continuum Stokes / eta_line
            + epsilon_continuum / eta_line

        where

        .. math::
            K_tau_line = K / eta_line = Kʹ / eta_lineʹ

            epsilon_tau_line = epsilon / eta_line = epsilonʹ / eta_lineʹ

        here, xʹ denotes x / N, where N is atomic concentration.



        dStokes/dtau_line = - K_tau_line * Stokes + epsilon_tau_line - K_continuum Stokes / eta_line +
        epsilon_continuum / eta_line

        if we now put K_line=epsilon_lin=0 (away from the line), we should get only the continuum:
        dStokes/dtau_line = - K_continuum Stokes / eta_line + epsilon_continuum / eta_line

        if we introduce eta_LC = eta_line / eta_continuum:

        dStokes/dtau_line = - K_continuum Stokes / eta_line + epsilon_continuum / eta_line =
        = - K_continuum Stokes / eta_continuum / eta_LC + epsilon_continuum /eta_continuum / eta_LC =
        = 1/eta_LC * (dStokes/dtau_continuum) by construction

        now, in case of unpolarized continuum:
        K_continuum = diag(eta_continuum) => K_continuum / eta_continuum = 1
        epsilon_continuum = eta_continuum * BP(T) * eI => epsilon_continuum / eta_continuum = BP(T) * eI

        where eI is the 0th Stokes component ort.

        Therefore, we can conclude:

        dStokes/dtau_line = - K_tau_line * Stokes + epsilon_tau_line
                            - Stokes / eta_LC + BP(T) eI / eta_LC

        In terms of dtau_continuum we then have:

        dStokes/dtau_continuum = - K_tau_line * Stokes * eta_LC + epsilon_tau_line * eta_LC - Stokes + BP(T) eI
        With a boundary condition of
        Stokes[tau->+inf] -> BP(T0)

        We can normalize the Stokes on BP(T0):
        dStokes/dtau_continuum = - K_tau_line * Stokes * eta_LC + epsilon_tau_line * eta_LC / BP(T0)
                                 - Stokes + BP(T)/BP(T0) eI

        With a boundary condition of
        Stokes[tau->+inf] -> 1
        """
        if initial_stokes is not None:
            assert len(initial_stokes.nu) == len(
                self.nu
            ), "Initial Stokes vector must have the same frequency grid as the slab"
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = initial_stokes.I
            self.initial_stokes[:, 1, 0] = initial_stokes.Q
            self.initial_stokes[:, 2, 0] = initial_stokes.U
            self.initial_stokes[:, 3, 0] = initial_stokes.V
        stokes = self.initial_stokes

        # Solve statistical equilibrium equations
        if isinstance(self.see, MultiTermAtomSEE):
            rho = self.see.get_solution()
        else:
            rho = self.see.get_solution(atmosphere_parameters=self.atmosphere_parameters)

        # Compute radiative transfer coefficients
        rtc = self.rte.calculate_all_coefficients(
            atmosphere_parameters=self.atmosphere_parameters,
            angles=self.angles,
            rho=rho,
        )

        # DELO method: S=K^-1 * epsilon, expM=expm(K*dtau), new_stokes = S + expM * (stokes - S)
        # Get normalized line coefficients
        K_line_norm = rtc.K_tau()  # [Nν, 4, 4] - NORMALIZED
        epsilon_line_norm = rtc.epsilon_tau()[:, :, 0]  # [Nν, 4] - NORMALIZED

        # Get the normalization scale factor
        scale_factor = rtc._eta_tau_scale  # Physical units → normalized

        # Scale continuum opacity to normalized units
        continuum_opacity_norm = self.continuum_opacity / scale_factor

        # Add continuum to NORMALIZED K
        K_total_norm = K_line_norm.copy()
        K_total_norm[:, 0, 0] += continuum_opacity_norm

        # For epsilon: ε_total = ε_line + χ_cont × S_cont
        # But ε_line is already normalized! Need to normalize χ_cont × S_cont too
        continuum_emissivity_physical = self.continuum_opacity * self.continuum_source_function
        continuum_emissivity_norm = continuum_emissivity_physical / scale_factor

        epsilon_total_norm = epsilon_line_norm.copy()
        epsilon_total_norm[:, 0] += continuum_emissivity_norm

        # Now everything is normalized, use self.tau
        S = np.stack(
            [
                (np.linalg.solve(K, eps) if np.linalg.cond(K) < 1e12 else (np.linalg.pinv(K) @ eps))
                for K, eps in zip(K_total_norm, epsilon_total_norm)
            ]
        )

        expM = np.stack([expm(-K * self.tau) for K in K_total_norm])
        stokes = S[:, :, np.newaxis] + np.einsum("nij,njk->nik", expM, stokes - S[:, :, np.newaxis])

        return Stokes(
            nu=self.nu,
            I=real(stokes[:, 0, 0]),
            Q=real(stokes[:, 1, 0]),
            U=real(stokes[:, 2, 0]),
            V=real(stokes[:, 3, 0]),
        )
