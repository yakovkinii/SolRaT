import logging
from typing import Dict

from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.general import half_int_to_str


class LevelRegistry:
    r"""
    Registry for atomic levels in the Multi-Level atom model.

    A level is identified by :math:`(\alpha, J)`, where :math:`\alpha` is a label that
    encapsulates all inner quantum numbers, and
    :math:`J` is the total angular momentum.  Each level carries a Lande factor
    :math:`g_{\alpha J}` that is supplied directly by the user.

    Alternatively, :math:`g_{\alpha J}` can be calculated using the LS coupling.
    """

    def __init__(self):
        self.levels: Dict[str, "Level"] = {}

    @log_method
    def register_level(self, alpha: str, J: float, energy_cmm1: float, g: float):
        r"""
        Register a new level.

        :param alpha:  string label denoting all quantum numbers other than :math:`J`.
        :param J:  half-int total angular momentum.
        :param energy_cmm1:  level energy in :math:`[1/\mathrm{cm}]`.
        :param g:  Lande factor :math:`g_{\alpha J}`.
        """
        level_id = self.construct_level_id(alpha=alpha, J=J)
        assert level_id not in self.levels.keys(), f"Level {level_id} is already registered."
        level = Level(level_id=level_id, alpha=alpha, J=J, energy_cmm1=energy_cmm1, g=g)
        self.levels[level_id] = level
        logging.debug(f"Level registry: registered {level_id}")

    def register_level_LS_coupling(self, alpha: str, L: float, S: float, J: float, energy_cmm1: float):
        r"""
        Register a new level with :math:`g_{\alpha J}` calculated using LS coupling.

        :param alpha:  string label denoting all quantum numbers other than :math:`J`.
        :param L:  half-int orbital momentum.
        :param S:  half-int spin.
        :param J:  half-int total angular momentum.
        :param energy_cmm1:  level energy in :math:`[1/\mathrm{cm}]`.
        """
        level_id = self.construct_level_id(alpha=alpha, J=J)
        assert level_id not in self.levels.keys(), f"Level {level_id} is already registered."
        if J == 0:
            g = 0
        else:
            g = 1 + 0.5 * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / J / (J + 1)
        level = Level(level_id=level_id, alpha=alpha, J=J, energy_cmm1=energy_cmm1, g=g)
        self.levels[level_id] = level
        logging.debug(f"Level registry: registered {level_id} (LS coupling)")

    @staticmethod
    def construct_level_id(alpha: str, J: float) -> str:
        """
        Construct a unique level ID.
        """
        return f"{alpha}_J={half_int_to_str(J)}"

    def get_level(self, alpha: str, J: float) -> "Level":
        r"""
        Get a level from :math:`\alpha, J`.
        """
        level_id = self.construct_level_id(alpha=alpha, J=J)
        assert level_id in self.levels.keys(), f"Level {level_id} is not registered."
        return self.levels[level_id]


class Level:
    r"""
    A single Multi-Level atom level :math:`(\alpha, J)`.

    :param level_id:  unique level ID.
    :param alpha:  string label denoting all quantum numbers other than :math:`J`.
    :param J:  half-int total angular momentum.
    :param energy_cmm1:  level energy in :math:`[1/\mathrm{cm}]`.
    :param g:  Lande factor :math:`g_{\alpha J}`.
    """

    def __init__(self, level_id: str, alpha: str, J: float, energy_cmm1: float, g: float):
        self.level_id: str = level_id
        self.alpha: str = alpha
        self.J: float = J
        self.energy_cmm1: float = energy_cmm1
        self.g: float = g

    def __hash__(self):
        return hash((self.level_id, self.alpha, self.J, self.energy_cmm1, self.g))
