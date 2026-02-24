import numpy as np


def pseudo_hash(stokesI, stokesQ, stokesU, stokesV):
    """
    Construct a pseudo-unique, likely non-trivial real float hash from 4 complex arrays
    """
    result = stokesI[::2] + stokesI[1::2] * 0.7
    result += stokesQ[::2] * 0.97 + stokesQ[1::2] * 0.77
    result += stokesU[::2] * 0.87 + stokesU[1::2] * 0.57
    result += stokesV[::2] * 0.77 + stokesV[1::2] * 0.37
    result = np.real(result) + np.imag(result) * 0.75
    return np.mean(result)
