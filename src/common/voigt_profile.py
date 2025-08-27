import numpy as np
from numpy import exp


def _voigt(nu: np.ndarray, a: float) -> np.ndarray:
    """
    complex Voigt profile at relative frequency nu with damping factor a.
    Reference: Humlíček, J. (1982). JQSRT, doi:10.1016/0022-4073(82)90078-4

    Note: HAZEL2 for some reason uses an expansion in terms of the Dawson's integral for a < 1e-3, see
    https://github.com/aasensio/hazel2/blob/master/src/hazel/maths.f90#L1076.
    They motivate it by some numerical instabilities near a->0.

    I didn't find any such issues here in SolRaT, and the Humlíček expansion approach matches the results from
    Dawson's integral expansion very closely, so I decided to use the Humlíček expansion for all a.
    """

    t = a - 1j * nu
    s = abs(nu) + a
    u = t * t

    if s >= 15:
        return t * 0.5641896 / (0.5 + u)
    if s >= 5.5:
        return t * (1.410474 + u * 0.5641896) / (0.75 + u * (3 + u))
    if a >= 0.195 * abs(nu) - 0.176:
        return (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))) / (
            16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )
    w4 = t * (
        36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419)))))
    )
    v4 = 32066.6 - u * (
        24322.84 - u * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u)))))
    )
    return exp(u) - w4 / v4


voigt = np.vectorize(_voigt)
