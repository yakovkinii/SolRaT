from typing import Dict

from solrat.atom_model.multi_level_atom_model.object.level_registry import Level
from solrat.atom_model.multi_level_atom_model.utility.einstein_coefficients import (
    b_lu_from_b_ul_multi_level_atom,
    b_ul_from_a_ul_multi_level_atom,
)
from solrat.atom_model.shared.utility.functions import energy_cmm1_to_frequency_sm1
from solrat.engine.functions.decorators import log_method


class TransitionRegistry:
    r"""
    Registry for radiative transitions between levels :math:`(\alpha_l J_l) \leftrightarrow (\alpha_u J_u)`.

    Only pure E1 transitions are supported.
    """

    def __init__(self):
        self.transitions: Dict[str, "Transition"] = {}

    def einstein_b_lu(self, transition_id: str) -> float:
        return self.transitions[transition_id].einstein_b_lu

    def einstein_b_ul(self, transition_id: str) -> float:
        return self.transitions[transition_id].einstein_b_ul

    def einstein_a_ul(self, transition_id: str) -> float:
        return self.transitions[transition_id].einstein_a_ul

    @log_method
    def register_transition(
        self,
        level_upper: Level,
        level_lower: Level,
        einstein_a_ul_sm1: float,
    ):
        r"""
        :param level_upper:  upper Level instance.
        :param level_lower:  lower Level instance.
        :param einstein_a_ul_sm1:  spontaneous emission Einstein coefficient :math:`A_{ul}` in :math:`[1/s]`.
        """
        transition_id = level_upper.level_id + "->" + level_lower.level_id
        assert transition_id not in self.transitions.keys()
        nu_ul = energy_cmm1_to_frequency_sm1(level_upper.energy_cmm1 - level_lower.energy_cmm1)
        b_ul = b_ul_from_a_ul_multi_level_atom(a_ul_sm1=einstein_a_ul_sm1, nu_ul=nu_ul)
        b_lu = b_lu_from_b_ul_multi_level_atom(b_ul=b_ul, Ju=level_upper.J, Jl=level_lower.J)

        self.transitions[transition_id] = Transition(
            transition_id=transition_id,
            level_upper=level_upper,
            level_lower=level_lower,
            einstein_a_ul=einstein_a_ul_sm1,
            einstein_b_ul=b_ul,
            einstein_b_lu=b_lu,
        )

    def is_transition_registered(self, level_upper: Level, level_lower: Level) -> bool:
        transition_id = level_upper.level_id + "->" + level_lower.level_id
        return transition_id in self.transitions.keys()

    def get_transition(self, level_upper: Level, level_lower: Level) -> "Transition":
        transition_id = level_upper.level_id + "->" + level_lower.level_id
        return self.transitions[transition_id]


class Transition:
    r"""
    Radiative transition :math:`(\alpha_u J_u) \to (\alpha_l J_l)`.

    :param transition_id:  unique transition ID.
    :param level_upper:  upper Level instance.
    :param level_lower:  lower Level instance.
    :param einstein_a_ul:  Einstein :math:`A_{ul}` :math:`[1/s]`.
    :param einstein_b_ul:  Einstein :math:`B_{ul}` :math:`[\mathrm{cm}^2/\mathrm{erg}/s]`.
    :param einstein_b_lu:  Einstein :math:`B_{lu}` :math:`[\mathrm{cm}^2/\mathrm{erg}/s]`.
    """

    def __init__(
        self,
        transition_id: str,
        level_upper: Level,
        level_lower: Level,
        einstein_a_ul: float,
        einstein_b_ul: float,
        einstein_b_lu: float,
    ):
        self.transition_id = transition_id
        self.level_upper: Level = level_upper
        self.level_lower: Level = level_lower
        self.einstein_a_ul: float = einstein_a_ul
        self.einstein_b_ul: float = einstein_b_ul
        self.einstein_b_lu: float = einstein_b_lu

    def get_mean_transition_frequency_sm1(self) -> float:
        r"""
        Transition frequency :math:`\nu_{ul} = c (E_u - E_l)`.

        Named ``mean`` for interface compatibility with the multi-term implementation.
        """
        return energy_cmm1_to_frequency_sm1(self.level_upper.energy_cmm1 - self.level_lower.energy_cmm1)
