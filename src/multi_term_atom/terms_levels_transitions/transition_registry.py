from typing import Dict

from src.common.functions import energy_cmm1_to_frequency_hz
from src.multi_term_atom.physics.einstein_coefficients import (
    b_lu_from_b_ul_multi_term_atom,
    b_ul_from_a_ul_multi_term_atom,
)
from src.multi_term_atom.terms_levels_transitions.level_registry import Term


class TransitionRegistry:
    def __init__(self):
        self.transitions: Dict[str, "Transition"] = {}

    def register_transition(
        self,
        term_upper: Term,
        term_lower: Term,
        einstein_a_ul: float,
        einstein_b_ul: float,
        einstein_b_lu: float,
    ):
        """
        Einstein coefficients are on betaLS->betaLS level, so need to sum over J in advance.
        A_ul = spontaneous emission
        B_ul = stimulated emission
        B_lu = absorption
        """
        assert term_lower.S == term_upper.S, "Spin of upper and lower terms must be the same"

        transition_id = term_upper.term_id + "->" + term_lower.term_id
        assert transition_id not in self.transitions.keys()
        transition = Transition(
            transition_id=transition_id,
            term_upper=term_upper,
            term_lower=term_lower,
            einstein_a_ul=einstein_a_ul,
            einstein_b_ul=einstein_b_ul,
            einstein_b_lu=einstein_b_lu,
        )
        self.transitions[transition_id] = transition

    def register_transition_from_a_ul(
        self,
        term_upper: Term,
        term_lower: Term,
        einstein_a_ul_sm1: float,
    ):
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
        )
        self.transitions[transition_id] = transition

    def is_transition_registered(
        self,
        term_upper: Term,
        term_lower: Term,
    ):
        transition_id = term_upper.term_id + "->" + term_lower.term_id
        return transition_id in self.transitions.keys()

    def get_transition(
        self,
        term_upper: Term,
        term_lower: Term,
    ):
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
    ):
        """
        Transition coefficients are registered between terms, not levels.
        """
        assert term_lower.S == term_lower.S
        self.transition_id = transition_id
        self.term_upper: Term = term_upper
        self.term_lower: Term = term_lower
        self.einstein_a_ul: float = einstein_a_ul
        self.einstein_b_ul: float = einstein_b_ul
        self.einstein_b_lu: float = einstein_b_lu

    def get_mean_transition_frequency_sm1(self):
        return energy_cmm1_to_frequency_hz(
            self.term_upper.get_mean_energy_cmm1() - self.term_lower.get_mean_energy_cmm1()
        )
