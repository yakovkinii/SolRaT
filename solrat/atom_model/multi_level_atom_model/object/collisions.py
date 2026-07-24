import logging
from typing import Dict

_NOT_VALIDATED_WARNING = (
    "ParametrizedCollisions: collisional rates in the multi-level SEE are a new, not-yet-validated "
    "feature (parametrized inelastic/superelastic transfer and elastic depolarizing rates; "
    "LL04 Sec. 7.13). Treat results that use collisions as experimental until the TB1999 benchmark "
    "reproduction is in place."
)


class ParametrizedCollisions:
    r"""
    Parametrized collisional rates for the multi-level SEE (LL04 Sec. 7.13).

    The user supplies, per radiative transition, the collisional (superelastic) de-excitation
    rate :math:`C_{ul}` [1/s] connecting the upper and lower level of that transition; the
    inelastic excitation rate :math:`C_{lu}` is obtained from the Einstein-Milne detailed-balance
    relation (LL04 eq. 7.98). Elastic depolarizing rates :math:`D^{(K)}` [1/s] (K >= 1;
    LL04 eq. 7.102) are supplied per level. All rates default to zero (no collisions).

    Rate-transfer multipole components are taken K-independent, :math:`C^{(K)} = C^{(0)}` (the
    K > 0 components require a detailed atom-collider model, LL04 App. 4); this is a documented
    parametrization, not an exact result.
    """

    def __init__(self):
        r"""
        Create an empty collisional-rate set (all rates zero until set explicitly).
        """
        self._deexcitation_rate_sm1: Dict[str, float] = {}
        self._depolarizing_rate_sm1: Dict[str, Dict[int, float]] = {}
        logging.warning(_NOT_VALIDATED_WARNING)

    def set_deexcitation_rate(self, transition_id: str, rate_sm1: float) -> None:
        r"""
        Set the collisional (superelastic) de-excitation rate :math:`C_{ul}` [1/s] for a
        transition (upper -> lower). The excitation rate is derived by detailed balance.
        """
        assert rate_sm1 >= 0, "deexcitation rate must be non-negative."
        self._deexcitation_rate_sm1[transition_id] = float(rate_sm1)

    def set_depolarizing_rate(self, level_id: str, K: int, rate_sm1: float) -> None:
        r"""
        Set the elastic depolarizing rate :math:`D^{(K)}` [1/s] for a level (K >= 1;
        :math:`D^{(0)} = 0`, populations are unaffected by elastic collisions).
        """
        assert K >= 1, "depolarizing rate is defined for K >= 1 (D^(0) = 0)."
        assert rate_sm1 >= 0, "depolarizing rate must be non-negative."
        self._depolarizing_rate_sm1.setdefault(level_id, {})[int(K)] = float(rate_sm1)

    def deexcitation_rate_sm1(self, transition_id: str) -> float:
        r"""
        Collisional de-excitation rate :math:`C_{ul}` [1/s] for a transition (0 if unset).
        """
        return self._deexcitation_rate_sm1.get(transition_id, 0.0)

    def depolarizing_rate_sm1(self, level_id: str, K: int) -> float:
        r"""
        Elastic depolarizing rate :math:`D^{(K)}` [1/s] for a level and rank K (0 if unset).
        """
        return self._depolarizing_rate_sm1.get(level_id, {}).get(int(K), 0.0)
