from typing import Union

import numpy as np
from numpy import real
from scipy.linalg import expm

from src.common.functions import get_planck_BP
from src.engine.functions.decorators import log_method
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.radiative_transfer_coefficients import (
    RadiativeTransferCoefficients,
)
from src.multi_term_atom.object.stokes import Stokes
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEELTE


class ConstantPropertySlabAtmosphere:
    def __init__(
        self,
        multi_term_atom_context: MultiTermAtomContext,
        radiation_tensor: RadiationTensor,
        line_delta_tau: float,
        continuum_delta_tau: float,
        angles: Angles,
        atmosphere_parameters: AtmosphereParameters,
    ):
        """
        A slab with constant atmospheric properties throughout its depth.
        Solves the radiative transfer equation using DELO method.

        .. math::
                dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

                eta_LC = tau_line / tau_continuum

                Stokes[tau->+inf] -> BP(T0)

        :param multi_term_atom_context: MultiTermAtomContext instance
        :param radiation_tensor: RadiationTensor instance (not used if SEE is MultiTermAtomSEELTE)
        :param line_delta_tau:  Optical thickness in line core. Avoid tiny/huge values for stability.
        :param continuum_delta_tau:  Optical thickness of continuum. Avoid tiny/huge values for stability.
        :param angles:  Angles instance
        :param atmosphere_parameters:  AtmosphereParameters instance

        Reference: modified (9.35-9.37)
        """
        assert line_delta_tau > 0, "Zero line_delta_tau is not supported within this formulation"
        assert continuum_delta_tau > 0, "Zero continuum_delta_tau is not supported within this formulation"
        self.multi_term_atom_context = multi_term_atom_context
        self.radiation_tensor = radiation_tensor
        self.line_delta_tau = line_delta_tau
        self.continuum_delta_tau = continuum_delta_tau
        self.angles = angles
        self.atmosphere_parameters = atmosphere_parameters
        self.nu = self.multi_term_atom_context.nu
        self.rte = MultiTermAtomRTE(
            level_registry=multi_term_atom_context.level_registry,
            transition_registry=multi_term_atom_context.transition_registry,
            nu=self.nu,
            precompute=False,
            j_constrained=self.multi_term_atom_context.j_constrained,
        )
        self.rtc: Union[RadiativeTransferCoefficients, None] = None  # Solved RTC are saved here

    @log_method
    def forward(self, initial_stokes: Stokes) -> Stokes:
        """
        Solve radiative transfer through the constant property slab using DELO method.
        .. math::
            dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

            eta_LC = tau_line / tau_continuum

            Stokes[tau->+inf] -> BP(T0)

        :param initial_stokes:  Initial Stokes vector that is entering the slab.

        Reference: modified (9.36)
        """

        # 1. Solve SEE for rho
        if isinstance(self.multi_term_atom_context.statistical_equilibrium_equations, MultiTermAtomSEELTE):
            rho = self.multi_term_atom_context.statistical_equilibrium_equations.get_solution(
                atmosphere_parameters=self.atmosphere_parameters
            )
        else:
            self.multi_term_atom_context.statistical_equilibrium_equations.fill_all_equations(
                atmosphere_parameters=self.atmosphere_parameters,
                radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(
                    chi_B=self.angles.chi_B, theta_B=self.angles.theta_B
                ),
            )
            rho = self.multi_term_atom_context.statistical_equilibrium_equations.get_solution()

        # Construct initial Stokes vector
        assert len(initial_stokes.nu) == len(
            self.nu
        ), "Initial Stokes vector must have the same frequency grid as the slab"
        stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
        stokes[:, 0, 0] = initial_stokes.I
        stokes[:, 1, 0] = initial_stokes.Q
        stokes[:, 2, 0] = initial_stokes.U
        stokes[:, 3, 0] = initial_stokes.V

        # Compute radiative transfer coefficients
        self.rtc = self.rte.calculate_all_coefficients(
            atmosphere_parameters=self.atmosphere_parameters,
            angles=self.angles,
            rho=rho,
        )

        #  dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

        K_tau_line = self.rtc.K_tau()  # [Nν, 4, 4]
        epsilon_tau_line = self.rtc.epsilon_tau()[:, :, 0]  # [Nν, 4]

        eta_LC = self.line_delta_tau / self.continuum_delta_tau

        # Add continuum
        K_tau_line[:, 0, 0] += 1 / eta_LC
        K_tau_line[:, 1, 1] += 1 / eta_LC
        K_tau_line[:, 2, 2] += 1 / eta_LC
        K_tau_line[:, 3, 3] += 1 / eta_LC
        epsilon_tau_line[:, 0] += get_planck_BP(nu_sm1=self.nu, T_K=self.atmosphere_parameters.temperature_K) / eta_LC

        # DELO method:
        # Solve dStokes/dtau = K*(Stokes-S), where
        # S=K^-1 * epsilon.
        # Solve by computing expM=expm(-K*dtau), new_stokes = S + expM * (stokes - S)

        S = np.stack(
            [
                (np.linalg.solve(K, eps) if np.linalg.cond(K) < 1e12 else (np.linalg.pinv(K) @ eps))
                for K, eps in zip(K_tau_line, epsilon_tau_line)
            ]
        )

        expM = np.stack([expm(-K * self.line_delta_tau) for K in K_tau_line])
        stokes = S[:, :, np.newaxis] + np.einsum("nij,njk->nik", expM, stokes - S[:, :, np.newaxis])

        return Stokes(
            nu=self.nu,
            I=real(stokes[:, 0, 0]),
            Q=real(stokes[:, 1, 0]),
            U=real(stokes[:, 2, 0]),
            V=real(stokes[:, 3, 0]),
        )
