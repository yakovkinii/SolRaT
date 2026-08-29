try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Union

import numpy as np
from numpy import sqrt

from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.multi_level_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.radiation_tensor import RadiationTensor
from solrat.atom_model.multi_level_atom_model.object.rho_matrix_builder import Rho, RhoMatrixBuilder
from solrat.atom_model.shared.utility.constants import kB_erg_Km1
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_erg
from solrat.engine.functions.decorators import log_method


class MultiLevelAtomSEELTE(BaseSEE):
    r"""
    Statistical Equilibrium Equations within the Multi-Level atom model - an LTE implementation.
    This class always outputs an LTE-distributed Rho tensor (thermal populations, no alignment),
    the multi-level counterpart of :class:`MultiTermAtomSEELTE`.

    :param level_registry:  LevelRegistry instance for the multi-level atom under study.
    """

    def __init__(self, level_registry: LevelRegistry):
        self.level_registry = level_registry
        self.matrix_builder: RhoMatrixBuilder = RhoMatrixBuilder(levels=list(self.level_registry.levels.values()))
        self._atmosphere_parameters: Union[AtmosphereParameters, None] = None

    @property
    def atmosphere_parameters(self) -> AtmosphereParameters:
        if self._atmosphere_parameters is None:
            raise RuntimeError("atmosphere_parameters has not been initialized")  # pragma: no cover
        return self._atmosphere_parameters

    @classmethod
    def from_model_config(cls, config: MultiLevelAtomConfig) -> Self:
        r"""
        Constructor from the model config.
        """
        logging.info("Constructing MultiLevelAtomSEELTE instance")
        return cls(level_registry=config.level_registry)

    @log_method
    def fill_all_equations(
        self,
        atmosphere_parameters: AtmosphereParameters,
        radiation_tensor_in_magnetic_frame: RadiationTensor,
    ):
        r"""
        Store the atmosphere parameters; the LTE SEE ignores the radiation field.

        :param atmosphere_parameters:  AtmosphereParameters instance carrying the temperature.
        :param radiation_tensor_in_magnetic_frame:  RadiationTensor instance (unused).
        """
        self._atmosphere_parameters = atmosphere_parameters

    @log_method
    def get_solution(self) -> Rho:
        r"""
        Return the LTE Rho solution: only the diagonal :math:`\rho^0_0(\alpha J)` populations are
        non-zero, Boltzmann-distributed and trace-normalized.

        .. math::

            \rho^0_0(\alpha J) &\sim \sqrt{2J+1}\, \mathrm{exp}(-E_J/kT)

            \Sigma \sqrt{2J+1}\, \rho^0_0(\alpha J) &= 1

        Reference: (LL04 3.108) (LL04 10.118)
        """
        logging.info("Applying LTE Statistical Equilibrium Equations solution")

        T = self.atmosphere_parameters.temperature_K
        min_energy = min([level.energy_cmm1 for level in self.level_registry.levels.values()])

        # Diagonal LTE populations rho^0_0(alpha J) per level, plus the normalization trace.
        rho_00 = {}
        trace = 0.0
        for level in self.level_registry.levels.values():
            E_erg = energy_cmm1_to_erg(level.energy_cmm1 - min_energy)
            value = np.sqrt(2 * level.J + 1) * np.exp(-E_erg / (kB_erg_Km1 * T))
            rho_00[level.level_id] = value
            trace += value * sqrt(2 * level.J + 1)

        # Single pass over all coherences: each is the trace-normalized diagonal value, or zero.
        rho = Rho(levels=list(self.level_registry.levels.values()))
        for index, (level_id, k, q) in self.matrix_builder.index_to_parameters.items():
            value = rho_00[level_id] / trace if (k == 0 and q == 0) else 0.0
            rho.set_from_level_id(level_id=level_id, K=k, Q=q, value=value)

        return rho
