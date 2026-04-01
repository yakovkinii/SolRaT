from abc import abstractmethod
from typing import Generic, TypeVar

import numpy as np
from typing_extensions import Self

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import AtmosphereParametersT
from solrat.atom_model.base_atom_model.object.config import ConfigT
from solrat.atom_model.base_atom_model.object.rho import RhoT
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.radiative_transfer_coefficients import RadiativeTransferCoefficients


class BaseRTE(Generic[ConfigT, AtmosphereParametersT, RhoT]):
    r"""
    Base class for radiative Transfer Equations.
    """

    @classmethod
    @abstractmethod
    def from_model_config(cls, config: ConfigT, nu: np.ndarray) -> Self:
        r"""
        Constructor from the model config.
        :param config: Model config
        :param nu: frequency array
        :return: :any:`BaseRTE` subclass instance
        """

    @abstractmethod
    def calculate_all_coefficients(
        self,
        atmosphere_parameters: AtmosphereParametersT,
        angles: Angles,
        rho: RhoT,
    ) -> RadiativeTransferCoefficients:
        r"""
        Compute all radiative transfer coefficients.

        :param atmosphere_parameters:  Atmosphere parameters that define RTE
        :param angles:  :any:`Angles` instance with LOS and magnetic field angles
        :param rho:  density tensor :math:`\rho^K_Q`
        :return: :any:`RadiativeTransferCoefficients` instance
        """


RTET = TypeVar("RTET", bound=BaseRTE)
