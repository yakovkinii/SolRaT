try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Union

import numpy as np
from numpy import sqrt

from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.multi_term_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.radiation_tensor import RadiationTensor
from solrat.atom_model.multi_term_atom_model.object.rho_matrix_builder import Rho, RhoMatrixBuilder
from solrat.atom_model.shared.utility.constants import kB_erg_Km1
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_erg
from solrat.engine.functions.decorators import log_method


class MultiTermAtomSEELTE(BaseSEE):
    r"""
    Statistical Equilibrium Equations within Multi-Term atom model - an LTE implementation.
    This class will always output an LTE-distributed Rho tensor.

    :param level_registry:  LevelRegistry instance for the multi-term atom under study.

    This is needed to be able to use SEELTE directly in nonLTE Radiative Transfer Equations.
    """

    def __init__(self, level_registry: LevelRegistry):
        self.level_registry = level_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(terms=list(self.level_registry.terms.values()))
        self._atmosphere_parameters: Union[AtmosphereParameters, None] = None

    @property
    def atmosphere_parameters(self) -> AtmosphereParameters:
        if self._atmosphere_parameters is None:
            raise RuntimeError("atmosphere_parameters has not been initialized")  # pragma: no cover
        return self._atmosphere_parameters

    @classmethod
    def from_model_config(cls, config: MultiTermAtomConfig) -> Self:
        r"""
        Constructor from the model config.
        """
        logging.info("Constructing MultiTermAtomSEELTE instance")
        return cls(
            level_registry=config.level_registry,
        )

    @log_method
    def fill_all_equations(
        self,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor_in_magnetic_frame: RadiationTensor,
    ):
        r"""
        Loop through all equations to construct the complete system of equations for rho.

        :param atmosphere_parameters:  AtmosphereParameters instance carrying the magnetic field and other variables.
        :param radiation_tensor_in_magnetic_frame:  RadiationTensor instance
        """
        self._atmosphere_parameters = atmosphere_parameters

    @log_method
    def get_solution(self) -> Rho:
        r"""
        Return LTE Rho solution.

        Assume Zeeman splitting does not cause energy shifts so large that they affect populations.

        Then:

        .. math::

            \rho^K_Q(J,Jʹ) &\sim \delta_{J,Jʹ}\delta_{K,0}\delta_{Q,0} \sqrt{2J+1} \mathrm{exp}(-E_J/kT)

            Tr[\rho^K_Q(J,Jʹ)] &= \Sigma \sqrt{2J+1} \rho^0_0(J,Jʹ) = 1

        Reference: (LL04 3.108) (LL04 10.118)
        """
        logging.info("Applying LTE Statistical Equilibrium Equations solution")

        T = self.atmosphere_parameters.temperature_K
        min_energy = min([level.energy_cmm1 for level in self.level_registry.levels.values()])

        # Diagonal LTE populations rho^0_0(J, J) per (term_id, J), plus the normalization trace.
        rho_00 = {}
        trace = 0.0
        for term in self.level_registry.terms.values():
            for level in term.levels:
                J = level.J
                E_erg = energy_cmm1_to_erg(level.energy_cmm1 - min_energy)
                value = np.sqrt(2 * J + 1) * np.exp(-E_erg / (kB_erg_Km1 * T))
                rho_00[(term.term_id, J)] = value
                trace += value * sqrt(2 * J + 1)

        # Single pass over all coherences: each is the trace-normalized diagonal value, or zero.
        # Setting each coherence exactly once (already normalized) keeps data and datarows consistent.
        rho = Rho(terms=list(self.level_registry.terms.values()))
        for index, (term_id, k, q, j, j_prime) in self.matrix_builder.index_to_parameters.items():
            if k == 0 and q == 0 and j == j_prime:
                value = rho_00.get((term_id, j), 0.0) / trace
            else:
                value = 0.0
            rho.set_from_term_id(term_id=term_id, K=k, Q=q, J=j, Jʹ=j_prime, value=value)

        return rho
