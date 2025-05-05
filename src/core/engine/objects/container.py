from typing import Dict, Union

import numpy as np

from src.core.engine.functions.general import half_int_to_str


class Container:
    def __init__(self):
        self.data: Dict[str, Union[float, np.ndarray]] = {}

    @staticmethod
    def get_key(**kwargs):
        return "_".join([f"{k}={v}" if isinstance(v, str) else f"{k}={half_int_to_str(v)}" for k, v in kwargs.items()])
