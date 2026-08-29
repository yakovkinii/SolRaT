from typing import TypeVar


class BaseAtmosphereParameters:
    r"""
    Base class for all atmosphere parameters needed in RTE/SEE. Subclasses are value objects: each
    defines :meth:`_key` (a tuple of its defining field values) and inherits hashing/equality by value,
    so a parameter set can key a cache on the atmosphere it describes (e.g. the RTE operator cache).
    """

    temperature_K: float

    def _key(self) -> tuple:
        r"""Value tuple identifying this parameter set; subclasses list their fields."""
        raise NotImplementedError  # pragma: no cover

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self._key() == other._key()


AtmosphereParametersT = TypeVar("AtmosphereParametersT", bound=BaseAtmosphereParameters)
