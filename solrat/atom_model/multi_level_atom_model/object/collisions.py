from typing import Dict

from numpy import exp

from solrat.atom_model.shared.utility.constants import h_erg_s, kB_erg_Km1


class ParametrizedCollisions:
    r"""
    Parametrized collisional rates for the SEE (LL04 Sec. 7.13), shared by the multi-level and the
    multi-term atoms.

    The user supplies the collisional (superelastic) de-excitation rate :math:`C_{ul}` [1/s]; the
    inelastic excitation rate :math:`C_{lu}` is obtained from the Einstein-Milne detailed-balance
    relation (LL04 eq. 7.98). For the multi-level atom the rate is set per transition (each transition
    is one fine-structure component). For the multi-term atom, whose transition is a whole multiplet,
    the rate is stored per fine-structure component :math:`(J_u, J_l)`: set one component with
    :meth:`set_deexcitation_rate_from_epsilon` (passing ``J_upper``/``J_lower``), or spread a single
    multiplet ``epsilon`` over all components with :meth:`fill_deexcitation_from_epsilon`. Elastic
    depolarizing rates :math:`D^{(K)}` [1/s] (K >= 1; LL04 eq. 7.102) are supplied per level. All rates
    default to zero (no collisions).

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

    def set_deexcitation_rate(self, transition_id: str, rate_sm1: float) -> None:
        r"""
        Set the collisional (superelastic) de-excitation rate :math:`C_{ul}` [1/s] for a
        transition (upper -> lower). The excitation rate is derived by detailed balance.
        """
        assert rate_sm1 >= 0, "deexcitation rate must be non-negative."
        self._deexcitation_rate_sm1[transition_id] = float(rate_sm1)

    @staticmethod
    def component_key(transition_id: str, J_upper: float, J_lower: float) -> str:
        r"""
        Storage key for one fine-structure component :math:`(J_u, J_l)` of a transition. The
        multi-term SEE stores and reads collisional de-excitation rates per component under this key;
        the multi-level atom (one component per transition) uses the bare ``transition_id``.
        """
        return f"{transition_id}|Ju={float(J_upper):.1f}|Jl={float(J_lower):.1f}"

    @staticmethod
    def _rate_from_epsilon(transition, epsilon: float, temperature_K: float) -> float:
        r"""
        :math:`C_{ul} = \frac{\epsilon}{1-\epsilon}\, A_{ul} / (1 - e^{-h\nu_0/kT})` (LL04 Sec. 7.13;
        TB1999 Sec. 2). Reads the transition through the interface shared by the multi-level and
        multi-term transitions (``get_mean_transition_frequency_sm1``, ``einstein_a_ul``).
        """
        assert 0.0 < epsilon < 1.0, "epsilon must be in (0, 1)."
        delta_e_erg = h_erg_s * transition.get_mean_transition_frequency_sm1()
        stimulated_correction = 1.0 - exp(-delta_e_erg / (kB_erg_Km1 * temperature_K))
        return epsilon / (1.0 - epsilon) * transition.einstein_a_ul / stimulated_correction

    def set_deexcitation_rate_from_epsilon(
        self, transition, epsilon: float, temperature_K: float, J_upper=None, J_lower=None
    ) -> None:
        r"""
        Set :math:`C_{ul}` from a photon destruction probability ``epsilon`` (LL04 Sec. 7.13; TB1999).

        Multi-level atom: call without ``J_upper``/``J_lower`` -- each transition is a single
        fine-structure component, keyed by its ``transition_id``. Multi-term atom: a transition is a
        whole multiplet, so pass ``J_upper`` and ``J_lower`` to set one fine-structure component (each
        component carries its own ``epsilon``, relative to the term :math:`A_{ul}`). To spread a single
        multiplet ``epsilon`` over all components automatically, use
        :meth:`fill_deexcitation_from_epsilon` instead.
        """
        rate_sm1 = self._rate_from_epsilon(transition, epsilon, temperature_K)
        is_multi_term = hasattr(transition, "term_upper")
        if is_multi_term and (J_upper is None or J_lower is None):
            raise ValueError(
                "For a multi-term (term-to-term) transition, pass J_upper and J_lower to set one "
                "fine-structure component, or call fill_deexcitation_from_epsilon to spread a single "
                "multiplet epsilon over all components."
            )
        if J_upper is None or J_lower is None:
            key = transition.transition_id
        else:
            key = self.component_key(transition.transition_id, J_upper, J_lower)
        self.set_deexcitation_rate(key, rate_sm1)

    def fill_deexcitation_from_epsilon(self, transition, epsilon: float, temperature_K: float) -> None:
        r"""
        Multi-term convenience: spread a single multiplet photon-destruction probability ``epsilon``
        over every fine-structure component :math:`(J_u, J_l)` of a term-to-term transition. The
        multiplet :math:`C_{ul}` is split uniformly over the :math:`n_l` lower-term levels,
        :math:`C_{ul}(J_u\!\to\! J_l)=C_{ul}/n_l`, so the total collisional de-excitation of each upper
        level stays :math:`C_{ul}` (the two-level interpretation). Reduces to a single
        :meth:`set_deexcitation_rate_from_epsilon` call for a one-J-per-term transition.
        """
        assert hasattr(transition, "term_upper"), (
            "fill_deexcitation_from_epsilon is for multi-term (term-to-term) transitions; for the "
            "multi-level atom use set_deexcitation_rate_from_epsilon."
        )
        lower_levels = transition.term_lower.levels
        c_ul = self._rate_from_epsilon(transition, epsilon, temperature_K) / len(lower_levels)
        for level_upper in transition.term_upper.levels:
            for level_lower in lower_levels:
                self.set_deexcitation_rate(
                    self.component_key(transition.transition_id, level_upper.J, level_lower.J), c_ul
                )

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
