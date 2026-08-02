from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np


class CompiledOperator:
    r"""
    A reduced frame captured as a dense operator, so the linear map it encodes can be re-applied
    without touching the Frame engine.

    After a frame is reduced to some multiplicand index columns plus a single evaluated ``coefficient``
    column (``reduce_partially`` -> ``to_coefficient``), the remaining computation
    ``result = sum_row coefficient[row] * value(row)`` is linear in the per-row ``value``. This class
    stores that constant part once: :attr:`keys` are the per-row multiplicand-index value tuples (in
    the order the frame was compiled with), and :attr:`coefficients` stacks the per-row coefficient
    arrays along a trailing axis (shape ``[*coefficient_shape, n_row]``). :meth:`multiply` then applies
    it against a callable evaluated at those keys.
    """

    def __init__(self, keys: List[Tuple], coefficients: np.ndarray):
        self.keys = keys
        self.coefficients = coefficients

    @classmethod
    def from_columns(
        cls, columns: Dict[str, np.ndarray], ordered_multiplicand_keys: List[str], coefficient_col: str = "coefficient"
    ) -> "CompiledOperator":
        keys = [tuple(values) for values in zip(*[columns[col] for col in ordered_multiplicand_keys])]
        if len(keys) == 0:
            raise ValueError(
                "Cannot build a CompiledOperator from a frame with no rows: there is nothing to "
                "contract against. This usually means no transition falls in the frequency range "
                "(check the frequency grid / transition cutoff)."
            )
        coefficients = np.stack([np.asarray(value, dtype=np.complex128) for value in columns[coefficient_col]], axis=-1)
        return cls(keys=keys, coefficients=coefficients)

    def multiply(self, value: Callable) -> np.ndarray:
        r"""
        Apply the operator against ``value``: ``result = sum_row coefficients[..., row] * value(*key)``,
        where ``value`` is called positionally with each row's key tuple (so the compile-time
        ``ordered_multiplicand_keys`` must match ``value``'s argument order). Equivalent to a Frame
        ``register_multiplication(value)`` followed by ``reduce`` over those keys.
        """
        values = np.array([value(*key) for key in self.keys], dtype=np.complex128)
        return np.tensordot(self.coefficients, values, axes=([self.coefficients.ndim - 1], [0]))


class OperatorCache:
    r"""
    Caches :class:`CompiledOperator` instances by the (hashable) arguments they are looked up with.

    ``get(*args)`` returns the operator stored for ``args`` (or ``None``); ``add(*args, operator=...)``
    stores it. The key is simply ``tuple(args)``, so any hashable arguments work -- e.g.
    ``cache.get(angles, atmosphere_parameters)`` keys by those two objects' values once they are
    value-hashable.

    Opt-in: caching is only active while :attr:`enabled` is ``True`` (``False`` by default). When
    disabled, ``get`` always returns ``None`` and ``add`` stores nothing, so the caller recomputes
    every call -- always correct, but only enable it when the cached operators stay valid (the same
    frequency grid and atomic data for this instance's lifetime).

    :param enabled:  whether caching is active.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._cache: Dict[Hashable, CompiledOperator] = {}

    def get(self, *key: Hashable) -> Optional[CompiledOperator]:
        return self._cache.get(key) if self.enabled else None

    def add(self, *key: Hashable, operator: CompiledOperator) -> None:
        if self.enabled:
            self._cache[key] = operator

    def clear(self) -> None:
        self._cache.clear()
