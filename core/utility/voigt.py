import numpy as np
from numpy import exp, sign

from core.utility.constant import sqrt_pi

H = 0.4
A1 = 2 / 3
A2 = 0.4
A3 = 2 / 7

C = [
    exp(-((1 * H) ** 2)),
    exp(-((3 * H) ** 2)),
    exp(-((5 * H) ** 2)),
    exp(-((7 * H) ** 2)),
    exp(-((9 * H) ** 2)),
    exp(-((11 * H) ** 2)),
]


def dawson(nu):
    """
    float Dawson's integral at frequency nu

    Reference: https://github.com/aasensio/hazel2/blob/master/src/hazel/maths.f90#L1034
    """
    if abs(nu) < 0.2:
        nu2 = nu**2
        return nu * (1 - A1 * nu2 * (1 - A2 * nu2 * (1 - A3 * nu2)))
    n0 = 2 * int(0.5 * abs(nu) / H)
    xp = abs(nu) - n0 * H
    e1 = exp(2 * xp * H)
    e2 = e1**2
    d1 = n0 + 1
    d2 = d1 - 2
    sum_i = 0
    for i in range(6):
        sum_i += C[i] * (e1 / d1 + 1 / d2 / e1)
        d1 = d1 + 2
        d2 = d2 - 2
        e1 = e2 * e1
    return 0.5641895835 * exp(-(xp**2)) * sign(nu) * sum_i


def _voigt(nu, a):
    """
    complex Voigt profile at relative frequency nu with damping factor a.

    Reference: https://github.com/aasensio/hazel2/blob/master/src/hazel/maths.f90#L1076
    See also: Humlicek (1982) JQSRT 27, 437
    """
    if a < 1e-3:
        return (
            exp(-(nu**2))
            + 2 * a / sqrt_pi * (2 * nu * dawson(nu) - 1)
            + 2j * dawson(nu) / sqrt_pi
            - 2j * a * nu * exp(-(nu**2))
        )

    t = a - 1j * nu
    s = abs(nu) + a
    u = t * t

    if s >= 15:
        return t * 0.5641896 / (0.5 + u)
    if s >= 5.5:
        return t * (1.410474 + u * 0.5641896) / (0.75 + u * (3 + u))
    if a >= 0.195 * abs(nu) - 0.176:
        return (
            16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))
        ) / (
            16.4955
            + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )
    w4 = t * (
        36183.31
        - u
        * (
            3321.9905
            - u
            * (
                1540.787
                - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419)))
            )
        )
    )
    v4 = 32066.6 - u * (
        24322.84
        - u
        * (
            9022.228
            - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u))))
        )
    )
    return exp(u) - w4 / v4


voigt = np.vectorize(_voigt)
