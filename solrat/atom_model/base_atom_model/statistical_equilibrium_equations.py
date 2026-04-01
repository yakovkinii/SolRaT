from abc import abstractmethod
from typing import Generic, TypeVar

from typing_extensions import Self

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import AtmosphereParametersT
from solrat.atom_model.base_atom_model.object.config import ConfigT
from solrat.atom_model.base_atom_model.object.radiation_tensor import RadiationTensorT
from solrat.atom_model.base_atom_model.object.rho import RhoT


class BaseSEE(Generic[ConfigT, AtmosphereParametersT, RadiationTensorT, RhoT]):
    r"""
    Base class for Statistical Equilibrium Equations
    """

    @classmethod
    @abstractmethod
    def from_model_config(cls, config: ConfigT) -> Self:
        r"""
        Constructor from the model config.
        :param config: model config
        :return: :any:`BaseSEE` subclass instance
        """

    @abstractmethod
    def fill_all_equations(
        self,
        atmosphere_parameters: AtmosphereParametersT,
        radiation_tensor_in_magnetic_frame: RadiationTensorT,
    ) -> None:
        r"""
        Loop through all equations to construct the complete system of equations for rho.

        :param atmosphere_parameters:  :any:`AtmosphereParameters` instance carrying the magnetic field
            and other variables.
        :param radiation_tensor_in_magnetic_frame:  :any:`RadiationTensor` instance
        """

    @abstractmethod
    def get_solution(self) -> RhoT:
        r"""
        Get the solution of the Statistical Equilibrium Equations.

        :return: :any:`Rho` instance
        """


SEET = TypeVar("SEET", bound=BaseSEE)
