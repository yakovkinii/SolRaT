from abc import abstractmethod

from typing_extensions import Self

from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.shared.object.angles import Angles


class BaseRadiationTensor:
    r"""
    Base class for Radiation tensor :math:`J^K_Q(\nu_{ul}`.
    """

    @classmethod
    @abstractmethod
    def from_model_config(cls, config: BaseConfig) -> Self:
        r"""
        Constructor from the model config.
        """

    @abstractmethod
    def rotate_to_magnetic_frame(self, angles: Angles) -> "BaseRadiationTensor":
        r"""
        Rotate :math:`J^K_Q(\nu_{ul}` to the magnetic reference frame.

        :param angles: :any:`Angles` instance with observation geometry.
        :return: :any:`BaseRadiationTensor` instance
        """
