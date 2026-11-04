r"""
Evolution operator of the polarized transfer equation, in closed form.

The propagation matrix of a spectral line has the structure of LL04 eq. (8.1),

.. math::

    \mathbf K = \eta_I\,\mathbf 1 + \mathbf K',\qquad
    \mathbf K' = \begin{pmatrix}
        0       & \eta_Q  & \eta_U  & \eta_V \\
        \eta_Q  & 0       & \rho_V  & -\rho_U \\
        \eta_U  & -\rho_V & 0       & \rho_Q \\
        \eta_V  & \rho_U  & -\rho_Q & 0
    \end{pmatrix}\,,

and the four eigenvalues of :math:`\mathbf K'` follow from
:math:`\lambda^4 - (\eta^2-\rho^2)\,\lambda^2 - (\bm\eta\cdot\bm\rho)^2 = 0`, giving
:math:`\pm\Lambda_+` and :math:`\pm i\Lambda_-` with

.. math::

    \Lambda_\pm^2 = \tfrac12\left(\sqrt{(\eta^2-\rho^2)^2 + 4(\bm\eta\cdot\bm\rho)^2} \pm (\eta^2-\rho^2)\right)\,.

Because the four eigenvalues are symmetric about zero, the Cayley-Hamilton expansion of
:math:`e^{-\mathbf K\,s}` has coefficients built from :math:`\cosh\Lambda_+s`, :math:`\sinh\Lambda_+s`,
:math:`\cos\Lambda_-s` and :math:`\sin\Lambda_-s` alone (Landi Degl'Innocenti & Landi Degl'Innocenti
1985, Solar Phys. 97, 239; LL04 Sec. 8.4). This module evaluates that expansion directly, which
replaces a numerical eigendecomposition per frequency point.

The result depends only on the structure of :math:`\mathbf K`, so it holds for every atomic
description and any number of levels or terms.
"""

import numpy as np

# Below this value of (Lambda_+ s)^2 + (Lambda_- s)^2 the coefficients are evaluated from their
# series, where the closed forms are 0/0. The series and the closed forms agree to full double
# precision on either side of it.
_DEGENERATE = 1e-8


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Elementwise ``numerator / denominator``, replaced by ``limit`` where the denominator vanishes."""
    degenerate = denominator < _DEGENERATE
    return np.where(degenerate, limit, numerator / np.where(degenerate, 1.0, denominator))


def propagation_matrix_invariants(K: np.ndarray):
    r"""
    The two eigenvalue scales of the propagation matrix.

    :param K: propagation matrices, shape ``[n, 4, 4]``.
    :return: tuple ``(eta_I, Lambda_plus, Lambda_minus)``, each of shape ``[n]``.
    """
    eta_I = K[:, 0, 0]
    eta_Q, eta_U, eta_V = K[:, 0, 1], K[:, 0, 2], K[:, 0, 3]
    rho_Q, rho_U, rho_V = K[:, 2, 3], K[:, 3, 1], K[:, 1, 2]

    eta2 = eta_Q**2 + eta_U**2 + eta_V**2
    rho2 = rho_Q**2 + rho_U**2 + rho_V**2
    eta_dot_rho = eta_Q * rho_Q + eta_U * rho_U + eta_V * rho_V

    difference = eta2 - rho2
    root = np.sqrt(difference**2 + 4.0 * eta_dot_rho**2)
    lambda_plus = np.sqrt(np.maximum(0.5 * (root + difference), 0.0))
    lambda_minus = np.sqrt(np.maximum(0.5 * (root - difference), 0.0))
    return eta_I, lambda_plus, lambda_minus


def evolution_operator(K: np.ndarray, ds: float) -> np.ndarray:
    r"""
    :math:`e^{-\mathbf K\,ds}` for a batch of propagation matrices, in closed form.

    The absorption factor :math:`e^{-\eta_I ds}` is folded into the hyperbolic terms, so that an
    optically thick step stays finite instead of overflowing through :math:`\cosh\Lambda_+ds`.

    :param K: propagation matrices, shape ``[n, 4, 4]``. A complex array is taken at its real part,
        as the propagation matrix of the transfer equation is real.
    :param ds: step length in the same units that make ``K ds`` dimensionless.
    :return: evolution operators, shape ``[n, 4, 4]``.
    """
    K = np.real(K)
    eta_I, lambda_plus, lambda_minus = propagation_matrix_invariants(K)
    absorption = eta_I * ds
    x = lambda_plus * ds
    y = lambda_minus * ds
    scale = x**2 + y**2

    # exp(-absorption) * cosh(x) and * sinh(x), written so that both exponents stay non-positive.
    upper = np.exp(x - absorption)
    lower = np.exp(-x - absorption)
    cosh_term = 0.5 * (upper + lower)
    sinh_term = 0.5 * (upper - lower)
    decay = np.exp(-absorption)
    cos_term = decay * np.cos(y)
    sin_term = decay * np.sin(y)

    # sinh(x)/x and sin(y)/y, both carrying the same absorption factor.
    sinhc = np.where(x > _DEGENERATE, sinh_term / np.where(x > _DEGENERATE, x, 1.0), decay * (1.0 + x**2 / 6.0))
    sinc = np.where(y > _DEGENERATE, sin_term / np.where(y > _DEGENERATE, y, 1.0), decay * (1.0 - y**2 / 6.0))

    # e^{-K ds} = a0 I + a1 N + a2 N^2 + a3 N^3 with N = -K' ds, from interpolating the exponential
    # on the four eigenvalues {+x, -x, +iy, -iy} of N.
    a0 = _safe_divide(cosh_term * y**2 + cos_term * x**2, scale, decay)
    a1 = _safe_divide(sinhc * y**2 + sinc * x**2, scale, decay)
    a2 = _safe_divide(cosh_term - cos_term, scale, 0.5 * decay)
    a3 = _safe_divide(sinhc - sinc, scale, decay / 6.0)

    identity = np.eye(4)
    N = -(K - eta_I[:, np.newaxis, np.newaxis] * identity) * ds
    N2 = N @ N
    N3 = N2 @ N
    coefficients = (a0, a1, a2, a3)
    matrices = (identity[np.newaxis], N, N2, N3)
    return sum(coefficient[:, np.newaxis, np.newaxis] * matrix for coefficient, matrix in zip(coefficients, matrices))


def evolution_and_interpolation_operators(K: np.ndarray, ds: float):
    r"""
    The pair used by the second-order DELO step: :math:`E = e^{-\mathbf K\,ds}` and
    :math:`P = (\mathbf K\,ds)^{-1}(\mathbf 1 - E)`.

    :math:`P` follows from :math:`E` by one batched linear solve, which replaces the explicit inverse.

    :param K: propagation matrices, shape ``[n, 4, 4]``.
    :param ds: step length.
    :return: tuple ``(E, P)``, each of shape ``[n, 4, 4]``.
    """
    K = np.real(K)
    E = evolution_operator(K, ds)
    identity = np.eye(4)
    K_ds = K * ds
    eta_I = K[:, 0, 0]
    # eta_I ds -> 0 leaves K ds singular; there P -> 1 - K ds / 2 to first order.
    thin = eta_I * ds < _DEGENERATE
    safe_K_ds = np.where(thin[:, np.newaxis, np.newaxis], identity, K_ds)
    P = np.linalg.solve(safe_K_ds, identity - E)
    return E, np.where(thin[:, np.newaxis, np.newaxis], identity - 0.5 * K_ds, P)
