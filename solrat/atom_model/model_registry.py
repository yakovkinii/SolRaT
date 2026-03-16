from typing import Generic, TypeVar, Union

from typing_extensions import Self

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import (
    BaseAtmosphereParameters,
)
from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.base_atom_model.object.radiation_tensor import (
    BaseRadiationTensor,
)
from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.multi_term_atom_model.data.FeI import get_Fe_I_5434_config
from solrat.atom_model.multi_term_atom_model.data.HeI import get_He_I_D3_config
from solrat.atom_model.multi_term_atom_model.data.MnI import get_Mn_I_5432_config
from solrat.atom_model.multi_term_atom_model.data.mock import get_mock_atom_config
from solrat.atom_model.multi_term_atom_model.data.NiI import get_Ni_I_5435_config
from solrat.atom_model.multi_term_atom_model.object.atmosphere_parameters import (
    AtmosphereParameters,
)
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import (
    MultiTermAtomConfig,
)
from solrat.atom_model.multi_term_atom_model.object.radiation_tensor import (
    RadiationTensor,
)
from solrat.atom_model.multi_term_atom_model.radiative_transfer_equations import (
    MultiTermAtomRTE,
)
from solrat.atom_model.multi_term_atom_model.statistical_equilibrium_equations import (
    MultiTermAtomSEE,
)
from solrat.atom_model.multi_term_atom_model_legacy.radiative_transfer_equations_legacy import (
    MultiTermAtomRTELegacy,
)
from solrat.atom_model.multi_term_atom_model_legacy.statistical_equilibrium_equations_legacy import (
    MultiTermAtomSEELegacy,
)
from solrat.atom_model.multi_term_atom_model_lte.object.radiation_tensor import (
    RadiationTensorLTE,
)
from solrat.atom_model.multi_term_atom_model_lte.statistical_equilibrium_equations import (
    MultiTermAtomSEELTE,
)

SEET = TypeVar("SEET", bound=BaseSEE)
RTET = TypeVar("RTET", bound=BaseRTE)
RadiationTensorT = TypeVar("RadiationTensorT", bound=BaseRadiationTensor)
AtmosphereParametersT = TypeVar("AtmosphereParametersT", bound=BaseAtmosphereParameters)
ConfigT = TypeVar("ConfigT", bound=BaseConfig)


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

        self.config: Union[ConfigT, None] = None

    def configure(self, config: ConfigT) -> Self:
        self.config = config
        return self


class Models:
    multi_term_atom = lambda: Model[  # noqa: E731
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

    multi_term_atom_legacy = lambda: Model[  # noqa: E731
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

    multi_term_atom_lte = lambda: Model[  # noqa: E731
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


# fmt: off
class PreconfiguredModels:
    multi_term_atom_HeID3 = lambda: Models.multi_term_atom().configure(config=get_He_I_D3_config())  # noqa: E731
    multi_term_atom_mock = lambda: Models.multi_term_atom().configure(config=get_mock_atom_config())  # noqa: E731
    multi_term_atom_mock_nofs = lambda: Models.multi_term_atom().configure(config=get_mock_atom_config(fine_structure=False))  # noqa: E501 E731
    multi_term_atom_mock_nofs_lte = lambda: Models.multi_term_atom_lte().configure(config=get_mock_atom_config(fine_structure=False))  # noqa: E501 E731
    multi_term_atom_legacy_mock = lambda: Models.multi_term_atom_legacy().configure(config=get_mock_atom_config())  # noqa: E501 E731
    multi_term_atom_legacy_mock_nofs = lambda: Models.multi_term_atom_legacy().configure(config=get_mock_atom_config(fine_structure=False))  # noqa: E501 E731
    multi_term_atom_lte_MnI_5432 = lambda: Models.multi_term_atom_lte().configure(config=get_Mn_I_5432_config())  # noqa: E501 E731
    multi_term_atom_lte_NiI_5435 = lambda: Models.multi_term_atom_lte().configure(config=get_Ni_I_5435_config())  # noqa: E501 E731
    multi_term_atom_lte_FeI_5434 = lambda: Models.multi_term_atom_lte().configure(config=get_Fe_I_5434_config())  # noqa: E501 E731
# fmt: on
