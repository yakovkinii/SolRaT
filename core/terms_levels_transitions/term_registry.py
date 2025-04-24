import logging
from typing import Dict, List

from core.base.python import half_int_to_str, triangular
from core.utility.constant import h


class TermRegistry:
    """
    Describes everything about the atom:
    levels, terms, transitions, their probability, etc.
    """

    def __init__(self):
        self.levels: Dict[str, Level] = {}
        self.terms: Dict[str, Term] = {}

    def register_term(self, beta: str, l: float, s: float, j: float, energy_cmm1: float):
        """
        beta: str - level ID
        l:half_int L
        s:half_int S
        j:half_int J
        """

        if energy_cmm1 < 50000 or energy_cmm1 > 150000:
            logging.warning(f"Received energy = {energy_cmm1} cm^-1. Please double check if the units are correct.")

        level_id = self.construct_level_id(beta=beta, l=l, s=s)
        self.add_level_if_needed(level_id=level_id, beta=beta, l=l, s=s)
        level = self.levels[level_id]

        term_id = self.construct_term_id(beta=beta, l=l, s=s, j=j)
        assert term_id not in self.terms.keys(), f"Term {term_id} is already registered."
        term = Term(level=level, term_id=term_id, j=j, energy_cmm1=energy_cmm1)

        level.register_term(term)
        self.terms[term_id] = term

    @staticmethod
    def construct_level_id(beta, l, s):
        return f"{beta}_l{half_int_to_str(l)}_s{half_int_to_str(s)}"

    @staticmethod
    def construct_term_id(beta, l, s, j):
        return f"{beta}_l{half_int_to_str(l)}_s{half_int_to_str(s)}_j{half_int_to_str(j)}"

    def add_level_if_needed(self, level_id: str, beta: str, l: float, s: float):
        if level_id not in self.levels.keys():
            logging.info(f"Creating level {level_id}")
            level = Level(level_id=level_id, beta=beta, l=l, s=s)
            self.levels[level_id] = level

    def validate(self):
        for level in self.levels.values():
            expected_j_values = triangular(level.l, level.s)
            actual_j_values = []
            for term in level.terms:
                if term.j in actual_j_values:
                    raise ValueError(f"Duplicate J values for level {level}: {[t.term_id for t in level.terms]}")
                actual_j_values.append(term.j)
            expected = set(expected_j_values)
            actual = set(actual_j_values)
            assert actual == expected, (
                f"Expected ({expected}) and actual ({actual}) j-values of terms "
                f"for level {level.level_id} do not match."
            )

    def get_term(self, level: "Level", j: float) -> "Term":
        term_id = self.construct_term_id(beta=level.beta, l=level.l, s=level.s, j=j)
        assert term_id in self.terms.keys(), f"Trying to get non-registered term {term_id}"
        return self.terms[term_id]

    def get_level(self, beta: str, l: float, s: float):
        level_id = self.construct_level_id(beta=beta, l=l, s=s)
        assert level_id in self.levels.keys()
        return self.levels[level_id]


class Level:
    """
    Level is {beta L S}
    """

    def __init__(self, level_id: str, beta: str, l: float, s: float):
        self.level_id: str = level_id
        self.beta: str = beta
        self.l: float = l
        self.s: float = s
        self.terms: List["Term"] = []

    def register_term(self, term: "Term"):
        assert term.beta == self.beta
        assert term.l == self.l
        assert term.s == self.s
        assert term not in self.terms
        self.terms.append(term)

    def get_term(self, J):
        """
        Get the term with the given J value.
        """
        for term in self.terms:
            if term.j == J:
                return term
        raise ValueError(f"Term with J={J} not found in level {self.level_id}.")


class Term:
    """
    Term is {beta L S J}
    """

    def __init__(self, level: "Level", term_id: str, j: float, energy_cmm1: float):
        assert abs(level.l - level.s) <= j <= level.l + level.s
        assert (level.l + level.s - j) % 1 == 0
        self.term_id: str = term_id
        self.beta: str = level.beta
        self.l: float = level.l
        self.s: float = level.s
        self.j: float = j
        self.energy_cmm1: float = energy_cmm1
        self.level: "Level" = level


def get_transition_frequency(energy_lower_cmm1: float, energy_upper_cmm1: float) -> float:
    # Todo needs to be checked
    return (energy_upper_cmm1 - energy_lower_cmm1) * h  # cm-1 -> Hz
