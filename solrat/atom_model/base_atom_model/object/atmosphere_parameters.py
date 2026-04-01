from typing import TypeVar


class BaseAtmosphereParameters:
    r"""
    Base class for all atmosphere parameters needed in RTE/SEE.
    """
    temperature_K: float


AtmosphereParametersT = TypeVar("AtmosphereParametersT", bound=BaseAtmosphereParameters)
