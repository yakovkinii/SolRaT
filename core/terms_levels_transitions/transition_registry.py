import logging
from typing import Dict

import numpy as np

from core.base.python import half_int_to_str
from core.terms_levels_transitions.term_registry import Level
from core.utility.constant import h, energy_cmm1_to_frequency_hz
from core.utility.einstein_coefficients import b_ul_from_a_two_level_atom, b_lu_from_b_ul_two_level_atom, \
    b_ul_from_a_two_term_atom, b_lu_from_b_ul_two_term_atom


class TransitionRegistry:
    def __init__(self):
        self.transitions: Dict[str, "Transition"] = {}

    def register_transition(
        self,
        level_upper: Level,
        level_lower: Level,
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

        transition_id = level_upper.level_id + "->" + level_lower.level_id
        assert transition_id not in self.transitions.keys()
        transition = Transition(
            transition_id=transition_id,
            level_upper=level_upper,
            level_lower=level_lower,
            einstein_a_ul=einstein_a_ul,
            einstein_b_ul=einstein_b_ul,
            einstein_b_lu=einstein_b_lu,
        )
        self.transitions[transition_id] = transition

    def register_transition_from_a_ul(
        self,
        level_upper: Level,
        level_lower: Level,
        einstein_a_ul_sm1: float,
    ):
        transition_id = level_upper.level_id + "->" + level_lower.level_id
        assert transition_id not in self.transitions.keys()
        nu_ul = energy_cmm1_to_frequency_hz(level_upper.get_mean_energy_cmm1() - level_lower.get_mean_energy_cmm1())

        b_ul = b_ul_from_a_two_term_atom(a_ul=einstein_a_ul_sm1, nu_ul=nu_ul)
        b_lu = b_lu_from_b_ul_two_term_atom(b_ul=b_ul, Lu=level_upper.l, Ll=level_lower.l)
        transition = Transition(
            transition_id=transition_id,
            level_upper=level_upper,
            level_lower=level_lower,
            einstein_a_ul=einstein_a_ul_sm1,
            einstein_b_ul=b_ul,
            einstein_b_lu=b_lu,
        )
        self.transitions[transition_id] = transition


    def is_transition_registered(
        self,
        level_upper: Level,
        level_lower: Level,
    ):
        transition_id = level_upper.level_id + "->" + level_lower.level_id
        return transition_id in self.transitions.keys()

    def get_transition(
        self,
        level_upper: Level,
        level_lower: Level,
    ):
        transition_id = level_upper.level_id + "->" + level_lower.level_id
        return self.transitions[transition_id]


class Transition:
    def __init__(
        self,
        transition_id: str,
        level_upper: Level,
        level_lower: Level,
        einstein_a_ul: float,
        einstein_b_ul: float,
        einstein_b_lu: float,
    ):
        assert level_lower.s == level_lower.s
        self.transition_id = transition_id
        self.level_upper: Level = level_upper
        self.level_lower: Level = level_lower
        self.einstein_a_ul: float = einstein_a_ul
        self.einstein_b_ul: float = einstein_b_ul
        self.einstein_b_lu: float = einstein_b_lu
