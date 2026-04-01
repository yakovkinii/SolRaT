try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

from abc import abstractmethod
from typing import Generic, TypeVar

from solrat.atom_model.base_atom_model.object.config import ConfigT
from solrat.atom_model.shared.object.angles import Angles


class BaseRadiationTensor(Generic[ConfigT]):
    r"""
    Base class for Radiation tensor :math:`J^K_Q(\nu_{ul}`.
    """

    @classmethod
    @abstractmethod
    def from_model_config(cls, config: ConfigT) -> Self:
        r"""
        Constructor from the model config.
        """
        raise NotImplementedError

    @abstractmethod
    def rotate_to_magnetic_frame(self, angles: Angles) -> "BaseRadiationTensor":
        r"""
        Rotate :math:`J^K_Q(\nu_{ul}` to the magnetic reference frame.

        :param angles: :any:`Angles` instance with observation geometry.
        :return: :any:`BaseRadiationTensor` instance
        """


RadiationTensorT = TypeVar("RadiationTensorT", bound=BaseRadiationTensor)
