from typing import TypeVar


class BaseConfig:
    r"""
    Base config class for atom models.
    """

    reference_lambda_A_air: float


ConfigT = TypeVar("ConfigT", bound=BaseConfig)
