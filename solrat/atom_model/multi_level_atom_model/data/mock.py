from typing import Optional

from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.engine.functions.decorators import log_function


@log_function
def get_mock_atom_config(  # pragma: no cover
    collisions: Optional[ParametrizedCollisions] = None,
) -> MultiLevelAtomConfig:
    r"""
    A :math:`J = 0 \to J = 1` resonance line as a multi-level atom, for testing.

    :param collisions:  Optional :class:`ParametrizedCollisions` to attach (default: collisionless).
    :return: :class:`MultiLevelAtomConfig` instance
    """
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="lower", J=0, energy_cmm1=0.0, g=1.0)
    # High transition energy puts the line deep in the Wien regime (h*nu0 / kT >> 1) at solar
    # temperatures, so the photon occupation number ~ exp(-h*nu0 / kT) -> 0 and stimulated emission
    # (RTE eta_S and SEE T_S / R_S) is negligible -- matching TB1999's neglect of stimulated emission.
    level_registry.register_level(alpha="upper", J=1, energy_cmm1=60_000.0, g=1.0)

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="upper", J=1),
        level_lower=level_registry.get_level(alpha="lower", J=0),
        einstein_a_ul_sm1=1e7,
    )

    return MultiLevelAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        atomic_mass_amu=4.0,
        reference_lambda_A_air=5000.0,
        collisions=collisions,
    )
