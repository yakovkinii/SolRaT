from typing import TypeVar


class BaseConfig:
    r"""
    Base config class for atom models.
    """
    reference_lambda_A: float


ConfigT = TypeVar("ConfigT", bound=BaseConfig)
