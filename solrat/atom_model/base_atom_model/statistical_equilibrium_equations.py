from abc import abstractmethod

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import (
    BaseAtmosphereParameters,
)
from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.base_atom_model.object.radiation_tensor import (
    BaseRadiationTensor,
)
from solrat.atom_model.base_atom_model.object.rho import BaseRho


class BaseSEE:
    r"""
    Base class for Statistical Equilibrium Equations
    """

    @classmethod
    @abstractmethod
    def from_model_config(cls, config: BaseConfig) -> "BaseSEE":
        r"""
        Constructor from the model config.
        :param config: model config
        :return: :any:`BaseSEE` subclass instance
        """

    @abstractmethod
    def fill_all_equations(
        self,
        atmosphere_parameters: BaseAtmosphereParameters,
        radiation_tensor_in_magnetic_frame: BaseRadiationTensor,
    ) -> None:
        r"""
        Loop through all equations to construct the complete system of equations for rho.

        :param atmosphere_parameters:  :any:`AtmosphereParameters` instance carrying the magnetic field
            and other variables.
        :param radiation_tensor_in_magnetic_frame:  :any:`RadiationTensor` instance
        """

    @abstractmethod
    def get_solution(self) -> BaseRho:
        r"""
        Get the solution of the Statistical Equilibrium Equations.

        :return: :any:`Rho` instance
        """
