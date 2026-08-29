import logging
from typing import Union

import numpy as np
from numpy import real
from scipy.linalg import expm

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import BaseAtmosphereParameters
from solrat.atom_model.base_atom_model.object.radiation_tensor import BaseRadiationTensor
from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.model_registry import Model
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.radiative_transfer_coefficients import RadiativeTransferCoefficients
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_planck_BP
from solrat.engine.functions.decorators import log_method


class ConstantPropertySlabAtmosphere:
    r"""
    A slab with constant atmospheric properties throughout its depth.
    Solves the radiative transfer equation using DELO method.

    .. math::

        dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

        eta_LC = tau_line / tau_continuum

        Stokes[tau->+inf] -> BP(T0)

    :param model: Model instance
    :param radiation_tensor: RadiationTensor instance (not used if SEE is MultiTermAtomSEELTE)
    :param line_delta_tau:  Optical thickness in line core. Avoid tiny/huge values for stability.
    :param continuum_delta_tau:  Optical thickness of continuum. Avoid tiny/huge values for stability.
    :param angles:  Angles instance
    :param atmosphere_parameters:  AtmosphereParameters instance

    Reference: modified (9.35-9.37)
    """

    def __init__(
        self,
        model: Model,
        radiation_tensor: BaseRadiationTensor,
        atmosphere_parameters: BaseAtmosphereParameters,
        angles: Angles,
        line_delta_tau: float,
        continuum_delta_tau: float,
    ):
        assert line_delta_tau > 0, "Zero line_delta_tau is not supported within this formulation"
        assert continuum_delta_tau > 0, "Zero continuum_delta_tau is not supported within this formulation"
        self.model = model
        self.radiation_tensor = radiation_tensor
        self.line_delta_tau = line_delta_tau
        self.continuum_delta_tau = continuum_delta_tau
        self.angles = angles
        self.atmosphere_parameters = atmosphere_parameters
        self.see = model.StatisticalEquilibriumEquations.from_model_config(config=model.config)
        self._rte: Union[BaseRTE, None] = None
        self._rtc: Union[RadiativeTransferCoefficients, None] = None  # Solved RTC are saved here

    @property
    def rte(self) -> BaseRTE:
        if self._rte is None:
            raise RuntimeError("rte has not been initialized")  # pragma: no cover
        return self._rte

    @property
    def rtc(self) -> RadiativeTransferCoefficients:
        if self._rtc is None:
            raise RuntimeError("rtc has not been initialized")  # pragma: no cover
        return self._rtc

    @log_method
    def forward(self, initial_stokes: Stokes) -> Stokes:
        r"""
        Solve radiative transfer through the constant property slab using DELO method.

        .. math::

            dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

            eta_LC = tau_line / tau_continuum

            Stokes[tau->+inf] -> BP(T0)

        :param initial_stokes:  Initial Stokes vector that is entering the slab.

        Reference: modified (9.36)
        """
        logging.info("Processing a single slab...")

        self.see.fill_all_equations(
            atmosphere_parameters=self.atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(angles=self.angles),
        )
        rho = self.see.get_solution()

        # Construct initial Stokes vector
        nu = initial_stokes.nu

        # Construct Stokes numpy vector
        stokes = np.zeros((len(nu), 4, 1), dtype=np.float64)
        stokes[:, 0, 0] = initial_stokes.I
        stokes[:, 1, 0] = initial_stokes.Q
        stokes[:, 2, 0] = initial_stokes.U
        stokes[:, 3, 0] = initial_stokes.V

        self._rte = self.model.RadiativeTransferEquations.from_model_config(config=self.model.config, nu=nu)

        # Compute radiative transfer coefficients
        self._rtc = self.rte.calculate_all_coefficients(
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
        epsilon_tau_line[:, 0] += (
            get_planck_BP(nu_sm1=nu, temperature_K=self.atmosphere_parameters.temperature_K) / eta_LC
        )

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

        logging.info("Completed processing a single slab")

        return Stokes(
            nu=nu,
            I=real(stokes[:, 0, 0]),
            Q=real(stokes[:, 1, 0]),
            U=real(stokes[:, 2, 0]),
            V=real(stokes[:, 3, 0]),
        )
