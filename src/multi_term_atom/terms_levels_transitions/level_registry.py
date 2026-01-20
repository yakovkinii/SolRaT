import logging
from typing import Dict, List, Union

from src.engine.functions.decorators import log_method, log_method_experimental
from src.engine.functions.general import half_int_to_str
from src.engine.functions.looping import triangular


class LevelRegistry:
    def __init__(self):
        """
        This class serves as a registry for all terms and levels.
        Level is {beta L S J}.
        Term is {beta L S}.
        """
        self.terms: Dict[str, Term] = {}
        self.levels: Dict[str, Level] = {}

    @log_method
    def register_level(self, beta: str, L: float, S: float, J: float, energy_cmm1: float):
        """
        Register a new level.

        :param beta: a string denoting the inner set of quantum numbers.
        :param L: half-int Orbital momentum.
        :param S: half-int Spin momentum.
        :param J: half-int Total momentum.
        :param energy_cmm1:  Level energy in [1/cm]
        """

        term_id = self.construct_term_id(beta=beta, L=L, S=S)
        self.register_term_if_needed(term_id=term_id, beta=beta, L=L, S=S)
        term = self.terms[term_id]

        level_id = self.construct_level_id(beta=beta, L=L, S=S, J=J)
        assert level_id not in self.levels.keys(), f"Level {level_id} is already registered."
        level = Level(term=term, level_id=level_id, J=J, energy_cmm1=energy_cmm1)

        term.register_level(level)
        self.levels[level_id] = level

    @staticmethod
    def construct_term_id(beta: str, L: float, S: float) -> str:
        """
        Construct a unique term ID
        """
        return f"{beta}_L={half_int_to_str(L)}_S={half_int_to_str(S)}"

    @staticmethod
    def construct_level_id(beta: str, L: float, S: float, J: float) -> str:
        """
        Construct a unique level ID
        """
        return f"{beta}_L={half_int_to_str(L)}_S={half_int_to_str(S)}_J={half_int_to_str(J)}"

    def register_term_if_needed(self, term_id: str, beta: str, L: float, S: float):
        """
        Register a new term (if not already registered).
        """
        if term_id not in self.terms.keys():
            logging.info(f"Level registry: Creating term {term_id}")
            term = Term(term_id=term_id, beta=beta, L=L, S=S)
            self.terms[term_id] = term

    @log_method
    def validate(self):
        """
        Perform a sanity check on all terms and levels:
        For each term, there should be levels with J from |L-S| to L+S
        """
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
        """
        Get level from term and J.
        """
        level_id = self.construct_level_id(beta=term.beta, L=term.L, S=term.S, J=J)
        assert level_id in self.levels.keys(), f"Trying to get non-registered level {level_id}"
        return self.levels[level_id]

    def get_term(self, beta: str, L: float, S: float) -> "Term":
        """
        Get term from beta L S
        """
        term_id = self.construct_term_id(beta=beta, L=L, S=S)
        assert term_id in self.terms.keys()
        return self.terms[term_id]


class Term:
    def __init__(
        self,
        term_id: str,
        beta: str,
        L: float,
        S: float,
    ):
        """
        Term is {beta L S}

        :param term_id: unique ID
        :param beta: a string denoting the inner set of quantum numbers.
        :param L: half-int Orbital momentum.
        :param S: half-int Spin momentum.
        """
        self.term_id: str = term_id
        self.beta: str = beta
        self.L: float = L
        self.S: float = S
        self.artificial_S_scale: Union[float, None] = None
        self.levels: List["Level"] = []

    @log_method_experimental
    def set_artificial_spin_scale(self, artificial_S_scale: float):
        """
        Set the artificial_S_scale.
        Caution: this is an experimental feature.

        The idea behind this mechanic is that the magnetic sensitivity of a line can be different from what
        LS coupling suggests. Therefore, this parameter can be used as a crude approach to model a different
        magnetic sensitivity by artificially scaling equation (3.3) as:
        H_B = mu0 * (Jz + scale * Sz) * B.
        For regular LS, scale=1.
        """
        self.artificial_S_scale = artificial_S_scale

    def register_level(self, level: "Level"):
        """
        Register a level to this term.
        """
        assert level.beta == self.beta
        assert level.L == self.L
        assert level.S == self.S
        assert level not in self.levels
        self.levels.append(level)

    def get_level(self, J) -> "Level":
        """
        Get the level with the given J value.
        """
        for level in self.levels:
            if level.J == J:
                return level
        raise ValueError(f"Level with J={J} not found in term {self.term_id}.")  # pragma: no cover

    def get_mean_energy_cmm1(self) -> float:
        """
        Get the non-weighted mean energy of the term.
        """
        total_energy = sum(level.energy_cmm1 for level in self.levels)
        return total_energy / len(self.levels)

    def get_max_energy_cmm1(self) -> float:
        """
        Get maximum level energy within this term
        :return:
        """
        return max(level.energy_cmm1 for level in self.levels)

    def get_min_energy_cmm1(self) -> float:
        """
        Get minimum level energy within this term
        :return:
        """
        return min(level.energy_cmm1 for level in self.levels)


class Level:
    def __init__(self, term: "Term", level_id: str, J: float, energy_cmm1: float):
        """
        Level is {beta L S J}

        :param term: Term instance
        :param level_id: Unique level ID
        :param J: half-int Total momentum.
        :param energy_cmm1: Level energy in [1/cm]
        """
        assert abs(term.L - term.S) <= J <= term.L + term.S
        assert (term.L + term.S - J) % 1 == 0
        self.level_id: str = level_id
        self.beta: str = term.beta
        self.L: float = term.L
        self.S: float = term.S
        self.J: float = J
        self.energy_cmm1: float = energy_cmm1
        self.term: "Term" = term
