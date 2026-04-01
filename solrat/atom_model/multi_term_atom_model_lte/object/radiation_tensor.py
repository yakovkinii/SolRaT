try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

from abc import abstractmethod

from solrat.atom_model.base_atom_model.object.config import BaseConfig
from solrat.atom_model.base_atom_model.object.radiation_tensor import BaseRadiationTensor
from solrat.atom_model.shared.object.angles import Angles


class RadiationTensorLTE(BaseRadiationTensor):
    r"""
    Base class for Radiation tensor :math:`J^K_Q(\nu_{ul}`.
    """

    def __init__(self):
        pass

    @classmethod
    def from_model_config(cls, config: BaseConfig) -> Self:
        r"""
        Constructor from the model config.
        """
        return cls()

    @abstractmethod
    def rotate_to_magnetic_frame(self, angles: Angles) -> Self:
        r"""
        Rotate JKQ to the magnetic reference frame.

        :param angles: Angles instance with observation geometry.
        """
        return self
