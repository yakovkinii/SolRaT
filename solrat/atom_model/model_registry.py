try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

import logging
from typing import Generic, Union

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import AtmosphereParametersT
from solrat.atom_model.base_atom_model.object.config import ConfigT
from solrat.atom_model.base_atom_model.object.radiation_tensor import RadiationTensorT
from solrat.atom_model.base_atom_model.radiative_transfer_equations import RTET
from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import SEET
from solrat.atom_model.multi_term_atom_model.data.FeI import get_Fe_I_5434_config
from solrat.atom_model.multi_term_atom_model.data.HeI import get_He_I_D3_config
from solrat.atom_model.multi_term_atom_model.data.MnI import get_Mn_I_5432_config
from solrat.atom_model.multi_term_atom_model.data.mock import get_mock_atom_config
from solrat.atom_model.multi_term_atom_model.data.NiI import get_Ni_I_5435_config
from solrat.atom_model.multi_term_atom_model.object.atmosphere_parameters import AtmosphereParameters
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.radiation_tensor import RadiationTensor
from solrat.atom_model.multi_term_atom_model.radiative_transfer_equations import MultiTermAtomRTE
from solrat.atom_model.multi_term_atom_model.statistical_equilibrium_equations import MultiTermAtomSEE
from solrat.atom_model.multi_term_atom_model_legacy.radiative_transfer_equations_legacy import MultiTermAtomRTELegacy
from solrat.atom_model.multi_term_atom_model_legacy.statistical_equilibrium_equations_legacy import (
    MultiTermAtomSEELegacy,
)
from solrat.atom_model.multi_term_atom_model_lte.object.radiation_tensor import RadiationTensorLTE
from solrat.atom_model.multi_term_atom_model_lte.statistical_equilibrium_equations import MultiTermAtomSEELTE
from solrat.engine.functions.decorators import log_method


class Model(Generic[SEET, RTET, RadiationTensorT, AtmosphereParametersT, ConfigT]):
    def __init__(
        self,
        StatisticalEquilibriumEquations: type[SEET],
        RadiativeTransferEquations: type[RTET],
        RadiationTensor: type[RadiationTensorT],
        AtmosphereParameters: type[AtmosphereParametersT],
        Config: type[ConfigT],
    ):
        self.StatisticalEquilibriumEquations: type[SEET] = StatisticalEquilibriumEquations
        self.RadiativeTransferEquations: type[RTET] = RadiativeTransferEquations
        self.RadiationTensor: type[RadiationTensorT] = RadiationTensor
        self.AtmosphereParameters: type[AtmosphereParametersT] = AtmosphereParameters
        self.Config: type[ConfigT] = Config

        self._config: Union[ConfigT, None] = None

    @property
    def config(self) -> ConfigT:
        if self._config is None:
            raise RuntimeError("config has not been initialized")
        return self._config

    @log_method
    def configure(self, config: ConfigT) -> Self:
        self._config = config
        return self


class Models:
    @staticmethod
    def multi_term_atom():
        logging.info("Creating Multi-Term Atom model")
        return Model[
            MultiTermAtomSEE,
            MultiTermAtomRTE,
            RadiationTensor,
            AtmosphereParameters,
            MultiTermAtomConfig,
        ](
            StatisticalEquilibriumEquations=MultiTermAtomSEE,
            RadiativeTransferEquations=MultiTermAtomRTE,
            RadiationTensor=RadiationTensor,
            AtmosphereParameters=AtmosphereParameters,
            Config=MultiTermAtomConfig,
        )

    @staticmethod
    def multi_term_atom_legacy():
        logging.info("Creating Multi-Term Atom Legacy model")
        return Model[
            MultiTermAtomSEELegacy,
            MultiTermAtomRTELegacy,
            RadiationTensor,
            AtmosphereParameters,
            MultiTermAtomConfig,
        ](
            StatisticalEquilibriumEquations=MultiTermAtomSEELegacy,
            RadiativeTransferEquations=MultiTermAtomRTELegacy,
            RadiationTensor=RadiationTensor,
            AtmosphereParameters=AtmosphereParameters,
            Config=MultiTermAtomConfig,
        )

    @staticmethod
    def multi_term_atom_lte():
        logging.info("Creating Multi-Term Atom LTE model")

        return Model[
            MultiTermAtomSEELTE,
            MultiTermAtomRTE,
            RadiationTensorLTE,
            AtmosphereParameters,
            MultiTermAtomConfig,
        ](
            StatisticalEquilibriumEquations=MultiTermAtomSEELTE,
            RadiativeTransferEquations=MultiTermAtomRTE,
            RadiationTensor=RadiationTensorLTE,
            AtmosphereParameters=AtmosphereParameters,
            Config=MultiTermAtomConfig,
        )


class PreconfiguredModels:
    @staticmethod
    def multi_term_atom_HeID3():
        return Models.multi_term_atom().configure(config=get_He_I_D3_config())

    @staticmethod
    def multi_term_atom_mock():
        return Models.multi_term_atom().configure(config=get_mock_atom_config())

    @staticmethod
    def multi_term_atom_mock_nofs():
        return Models.multi_term_atom().configure(config=get_mock_atom_config(fine_structure=False))

    @staticmethod
    def multi_term_atom_mock_nofs_lte():
        return Models.multi_term_atom_lte().configure(config=get_mock_atom_config(fine_structure=False))

    @staticmethod
    def multi_term_atom_legacy_mock():
        return Models.multi_term_atom_legacy().configure(config=get_mock_atom_config())

    @staticmethod
    def multi_term_atom_legacy_mock_nofs():
        return Models.multi_term_atom_legacy().configure(config=get_mock_atom_config(fine_structure=False))

    @staticmethod
    def multi_term_atom_lte_MnI_5432():
        return Models.multi_term_atom_lte().configure(config=get_Mn_I_5432_config())

    @staticmethod
    def multi_term_atom_lte_NiI_5435():
        return Models.multi_term_atom_lte().configure(config=get_Ni_I_5435_config())

    @staticmethod
    def multi_term_atom_lte_FeI_5434():
        return Models.multi_term_atom_lte().configure(config=get_Fe_I_5434_config())
