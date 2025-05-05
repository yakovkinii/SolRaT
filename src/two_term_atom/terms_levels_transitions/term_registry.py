import logging
from typing import Dict, List

from src.core.engine.functions.general import half_int_to_str
from src.core.engine.functions.looping import triangular


class TermRegistry:
    """
    Describes everything about the atom:
    levels, terms, transitions, their probability, etc.
    """

    def __init__(self):
        self.levels: Dict[str, Level] = {}
        self.terms: Dict[str, Term] = {}

    def register_term(self, beta: str, L: float, S: float, J: float, energy_cmm1: float):
        """
        beta: str - level ID
        l:half_int L
        s:half_int S
        j:half_int J
        """

        level_id = self.construct_level_id(beta=beta, L=L, S=S)
        self.add_level_if_needed(level_id=level_id, beta=beta, L=L, S=S)
        level = self.levels[level_id]

        term_id = self.construct_term_id(beta=beta, L=L, S=S, J=J)
        assert term_id not in self.terms.keys(), f"Term {term_id} is already registered."
        term = Term(level=level, term_id=term_id, J=J, energy_cmm1=energy_cmm1)

        level.register_term(term)
        self.terms[term_id] = term

    @staticmethod
    def construct_level_id(beta, L, S):
        return f"{beta}_L={half_int_to_str(L)}_S={half_int_to_str(S)}"

    @staticmethod
    def construct_term_id(beta, L, S, J):
        return f"{beta}_L={half_int_to_str(L)}_S={half_int_to_str(S)}_J={half_int_to_str(J)}"

    def add_level_if_needed(self, level_id: str, beta: str, L: float, S: float):
        if level_id not in self.levels.keys():
            logging.info(f"Term registry: Creating level {level_id}")
            level = Level(level_id=level_id, beta=beta, L=L, S=S)
            self.levels[level_id] = level

    def validate(self):
        for level in self.levels.values():
            expected_j_values = triangular(level.L, level.S)
            actual_j_values = []
            for term in level.terms:
                assert (
                    term.J not in actual_j_values
                ), f"Duplicate J values for level {level.level_id}: {term.J} in {[t.term_id for t in level.terms]}"
                actual_j_values.append(term.J)
            expected = set(expected_j_values)
            actual = set(actual_j_values)
            assert actual == expected, (
                f"Expected ({expected}) and actual ({actual}) j-values of terms "
                f"for level {level.level_id} do not match."
            )

    def get_term(self, level: "Level", J: float) -> "Term":
        term_id = self.construct_term_id(beta=level.beta, L=level.L, S=level.S, J=J)
        assert term_id in self.terms.keys(), f"Trying to get non-registered term {term_id}"
        return self.terms[term_id]

    def get_level(self, beta: str, L: float, S: float):
        level_id = self.construct_level_id(beta=beta, L=L, S=S)
        assert level_id in self.levels.keys()
        return self.levels[level_id]


class Level:
    """
    Level is {beta L S}
    """

    def __init__(self, level_id: str, beta: str, L: float, S: float):
        self.level_id: str = level_id
        self.beta: str = beta
        self.L: float = L
        self.S: float = S
        self.terms: List["Term"] = []

    def register_term(self, term: "Term"):
        assert term.beta == self.beta
        assert term.L == self.L
        assert term.S == self.S
        assert term not in self.terms
        self.terms.append(term)

    def get_term(self, J):
        """
        Get the term with the given J value.
        """
        for term in self.terms:
            if term.J == J:
                return term
        raise ValueError(f"Term with J={J} not found in level {self.level_id}.")

    def get_mean_energy_cmm1(self):
        """
        Get the mean energy of the level.
        """
        total_energy = sum(term.energy_cmm1 for term in self.terms)
        return total_energy / len(self.terms)


class Term:
    """
    Term is {beta L S J}
    """

    def __init__(self, level: "Level", term_id: str, J: float, energy_cmm1: float):
        assert abs(level.L - level.S) <= J <= level.L + level.S
        assert (level.L + level.S - J) % 1 == 0
        self.term_id: str = term_id
        self.beta: str = level.beta
        self.L: float = level.L
        self.S: float = level.S
        self.J: float = J
        self.energy_cmm1: float = energy_cmm1
        self.level: "Level" = level
