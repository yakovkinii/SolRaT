import logging
from typing import Dict, List

from src.core.engine.functions.general import half_int_to_str
from src.core.engine.functions.looping import triangular


class LevelRegistry:
    """
    Describes everything about the atom:
    terms, levels, transitions, their probability, etc.
    """

    def __init__(self):
        self.terms: Dict[str, Term] = {}
        self.levels: Dict[str, Level] = {}

    def register_level(self, beta: str, L: float, S: float, J: float, energy_cmm1: float):
        """
        beta: str - term ID
        l:half_int L
        s:half_int S
        j:half_int J
        """

        term_id = self.construct_term_id(beta=beta, L=L, S=S)
        self.add_term_if_needed(term_id=term_id, beta=beta, L=L, S=S)
        term = self.terms[term_id]

        level_id = self.construct_level_id(beta=beta, L=L, S=S, J=J)
        assert level_id not in self.levels.keys(), f"Level {level_id} is already registered."
        level = Level(term=term, level_id=level_id, J=J, energy_cmm1=energy_cmm1)

        term.register_level(level)
        self.levels[level_id] = level

    @staticmethod
    def construct_term_id(beta, L, S):
        return f"{beta}_L={half_int_to_str(L)}_S={half_int_to_str(S)}"

    @staticmethod
    def construct_level_id(beta, L, S, J):
        return f"{beta}_L={half_int_to_str(L)}_S={half_int_to_str(S)}_J={half_int_to_str(J)}"

    def add_term_if_needed(self, term_id: str, beta: str, L: float, S: float):
        if term_id not in self.terms.keys():
            logging.info(f"Level registry: Creating term {term_id}")
            term = Term(term_id=term_id, beta=beta, L=L, S=S)
            self.terms[term_id] = term

    def validate(self):
        for term in self.terms.values():
            expected_j_values = triangular(term.L, term.S)
            actual_j_values = []
            for level in term.levels:
                assert (
                    level.J not in actual_j_values
                ), f"Duplicate J values for term {term.term_id}: {level.J} in {[t.level_id for t in term.levels]}"
                actual_j_values.append(level.J)
            expected = set(expected_j_values)
            actual = set(actual_j_values)
            assert actual == expected, (
                f"Expected ({expected}) and actual ({actual}) j-values of levels "
                f"for term {term.term_id} do not match."
            )

    def get_level(self, term: "Term", J: float) -> "Level":
        level_id = self.construct_level_id(beta=term.beta, L=term.L, S=term.S, J=J)
        assert level_id in self.levels.keys(), f"Trying to get non-registered level {level_id}"
        return self.levels[level_id]

    def get_term(self, beta: str, L: float, S: float):
        term_id = self.construct_term_id(beta=beta, L=L, S=S)
        assert term_id in self.terms.keys()
        return self.terms[term_id]


class Term:
    """
    Term is {beta L S}
    """

    def __init__(self, term_id: str, beta: str, L: float, S: float):
        self.term_id: str = term_id
        self.beta: str = beta
        self.L: float = L
        self.S: float = S
        self.levels: List["Level"] = []

    def register_level(self, level: "Level"):
        assert level.beta == self.beta
        assert level.L == self.L
        assert level.S == self.S
        assert level not in self.levels
        self.levels.append(level)

    def get_level(self, J):
        """
        Get the level with the given J value.
        """
        for level in self.levels:
            if level.J == J:
                return level
        raise ValueError(f"Level with J={J} not found in term {self.term_id}.")  # pragma: no cover

    def get_mean_energy_cmm1(self):
        """
        Get the mean energy of the term.
        """
        total_energy = sum(level.energy_cmm1 for level in self.levels)
        return total_energy / len(self.levels)


class Level:
    """
    Level is {beta L S J}
    """

    def __init__(self, term: "Term", level_id: str, J: float, energy_cmm1: float):
        assert abs(term.L - term.S) <= J <= term.L + term.S
        assert (term.L + term.S - J) % 1 == 0
        self.level_id: str = level_id
        self.beta: str = term.beta
        self.L: float = term.L
        self.S: float = term.S
        self.J: float = J
        self.energy_cmm1: float = energy_cmm1
        self.term: "Term" = term
