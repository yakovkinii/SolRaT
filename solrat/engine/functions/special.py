from typing import TypeVar

import numpy as np

from solrat.engine.functions.decorators import log_function

FloatOrNDArrayT = TypeVar("FloatOrNDArrayT", float, np.ndarray)
IntOrFloatT = TypeVar("IntOrFloatT", int, float)
NumericT = TypeVar("NumericT", int, float, complex, np.ndarray)


@log_function
def pseudo_hash(stokesI, stokesQ, stokesU, stokesV):
    """
    Construct a pseudo-unique, likely non-trivial real float hash from 4 complex arrays
    """
    if stokesI.shape[0] % 2 == 1:
        stokesI = stokesI[:-1] + stokesI[-1]
        stokesQ = stokesQ[:-1] + stokesQ[-1]
        stokesU = stokesU[:-1] + stokesU[-1]
        stokesV = stokesV[:-1] + stokesV[-1]

    result = stokesI[::2] + stokesI[1::2] * 0.7
    result += stokesQ[::2] * 0.97 + stokesQ[1::2] * 0.77
    result += stokesU[::2] * 0.87 + stokesU[1::2] * 0.57
    result += stokesV[::2] * 0.77 + stokesV[1::2] * 0.37
    result = np.real(result) + np.imag(result) * 0.75
    return np.mean(result)
