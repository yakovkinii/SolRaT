from typing import Dict, List

from src.common.functions import energy_cmm1_to_frequency_hz
from src.multi_term_atom.physics.einstein_coefficients import (
    b_lu_from_b_ul_multi_term_atom,
    b_ul_from_a_ul_multi_term_atom,
)
from src.multi_term_atom.terms_levels_transitions.level_registry import Term


class TransitionRegistry:
    def __init__(self):
        """
        This class serves as a registry for all transitions.
        The transitions are defined as ones occurring between terms, not levels,
        so the Einstein coefficients need to be adjusted accordingly.

        Only pure E1 LS transitions are fully supported.
        """
        self.transitions: Dict[str, "Transition"] = {}

    def register_transition(
        self,
        term_upper: Term,
        term_lower: Term,
        einstein_a_ul_sm1: float,
        lower_J_constraint: List[float] = None,
        upper_J_constraint: List[float] = None,
    ):
        """
        Einstein coefficients are on betaLS->betaLS level, so need to sum over J in advance.

        :param term_upper: upper Term instance
        :param term_lower: lower Term instance
        :param einstein_a_ul_sm1: spontaneous emission in [1/s]
        :param lower_J_constraint:  constrain lower term J values in RTE to these values
        :param upper_J_constraint:  constrain upper term J values in RTE to these values
        """
        assert term_lower.S == term_upper.S, "Spin of upper and lower terms must be the same"

        transition_id = term_upper.term_id + "->" + term_lower.term_id
        assert transition_id not in self.transitions.keys()
        nu_ul = energy_cmm1_to_frequency_hz(term_upper.get_mean_energy_cmm1() - term_lower.get_mean_energy_cmm1())

        b_ul = b_ul_from_a_ul_multi_term_atom(a_ul_sm1=einstein_a_ul_sm1, nu_ul=nu_ul)
        b_lu = b_lu_from_b_ul_multi_term_atom(b_ul=b_ul, Lu=term_upper.L, Ll=term_lower.L)
        transition = Transition(
            transition_id=transition_id,
            term_upper=term_upper,
            term_lower=term_lower,
            einstein_a_ul=einstein_a_ul_sm1,
            einstein_b_ul=b_ul,
            einstein_b_lu=b_lu,
            lower_J_constraint=lower_J_constraint,
            upper_J_constraint=upper_J_constraint,
        )
        self.transitions[transition_id] = transition

    def is_transition_registered(
        self,
        term_upper: Term,
        term_lower: Term,
    ):
        """
        Checks if the transition between the terms is registered.
        """
        transition_id = term_upper.term_id + "->" + term_lower.term_id
        return transition_id in self.transitions.keys()

    def get_transition(
        self,
        term_upper: Term,
        term_lower: Term,
    ):
        """
        Get the transition between the two terms - assume it is registered.
        """
        transition_id = term_upper.term_id + "->" + term_lower.term_id
        return self.transitions[transition_id]


class Transition:
    def __init__(
        self,
        transition_id: str,
        term_upper: Term,
        term_lower: Term,
        einstein_a_ul: float,
        einstein_b_ul: float,
        einstein_b_lu: float,
        lower_J_constraint: List[float] = None,
        upper_J_constraint: List[float] = None,
    ):
        """
        Transition coefficients are registered between terms, not levels.


        :param transition_id:  Unique transition ID
        :param term_upper: upper Term instance
        :param term_lower: lower Term instance
        :param einstein_a_ul: Einstein coefficient Aul
        :param einstein_b_ul: Einstein coefficient Bul
        :param einstein_b_lu: Einstein coefficient Blu
        :param lower_J_constraint: constrain lower term J values in RTE to these values
        :param upper_J_constraint: constrain lower term J values in RTE to these values
        """
        assert term_lower.S == term_lower.S
        self.transition_id = transition_id
        self.term_upper: Term = term_upper
        self.term_lower: Term = term_lower
        self.einstein_a_ul: float = einstein_a_ul
        self.einstein_b_ul: float = einstein_b_ul
        self.einstein_b_lu: float = einstein_b_lu
        if lower_J_constraint is not None:
            for J in lower_J_constraint:
                assert J in [level.J for level in term_lower.levels]
        if upper_J_constraint is not None:
            for J in upper_J_constraint:
                assert J in [level.J for level in term_upper.levels]
        self.lower_J_for_RTE = lower_J_constraint
        self.upper_J_for_RTE = upper_J_constraint

    def get_mean_transition_frequency_sm1(self):
        """
        A crude approximation for the 'central' frequency of the transition.
        Should be used only in non-frequency-sensitive expressions like filling out Planck's function in LTE.
        """
        return energy_cmm1_to_frequency_hz(
            self.term_upper.get_mean_energy_cmm1() - self.term_lower.get_mean_energy_cmm1()
        )
