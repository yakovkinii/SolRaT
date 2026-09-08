from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.multi_term_atom_model.data.HeI import get_He_I_D3_config as get_multi_term_He_I_D3_config


def _is_allowed_e1_component(Ju: float, Jl: float) -> bool:
    return abs(Ju - Jl) <= 1 and not (Ju == 0 and Jl == 0)


_COMPONENT_A_UL_SM1 = {
    ("2p3", 0, "2s3", 1): 1.022e7,
    ("2p3", 1, "2s3", 1): 1.022e7,
    ("2p3", 2, "2s3", 1): 1.022e7,
    ("3p3", 0, "2s3", 1): 9.478e6,
    ("3p3", 1, "2s3", 1): 9.478e6,
    ("3p3", 2, "2s3", 1): 9.478e6,
    ("3s3", 1, "2p3", 0): 3.080e6,
    ("3s3", 1, "2p3", 1): 9.250e6,
    ("3s3", 1, "2p3", 2): 1.540e7,
    ("3d3", 1, "2p3", 0): 3.920e7,
    ("3d3", 2, "2p3", 1): 5.290e7,
    ("3d3", 1, "2p3", 1): 2.940e7,
    ("3d3", 3, "2p3", 2): 7.060e7,
    ("3d3", 2, "2p3", 2): 1.760e7,
    ("3d3", 1, "2p3", 2): 1.960e6,
}


def get_He_I_D3_config() -> MultiLevelAtomConfig:  # pragma: no cover
    r"""
    Direct multi-level analog of the built-in multi-term He I D3 atom.

    Each :math:`LSJ` level of the multi-term atom is registered as an independent multi-level level,
    with its Lande factor calculated in LS coupling. For each registered multi-term transition, all
    allowed electric-dipole :math:`J_u \to J_l` component branches are registered with their
    component Einstein :math:`A_{ul}` values.

    :return: :any:`MultiLevelAtomConfig` instance.
    """
    multi_term_config = get_multi_term_He_I_D3_config()

    level_registry = LevelRegistry()
    for term in multi_term_config.level_registry.terms.values():
        for level in term.levels:
            level_registry.register_level_LS_coupling(
                alpha=term.term_id,
                L=term.L,
                S=term.S,
                J=level.J,
                energy_cmm1=level.energy_cmm1,
            )

    transition_registry = TransitionRegistry()
    for transition in multi_term_config.transition_registry.transitions.values():
        upper_term = transition.term_upper
        lower_term = transition.term_lower
        for upper_level in upper_term.levels:
            for lower_level in lower_term.levels:
                if not _is_allowed_e1_component(Ju=upper_level.J, Jl=lower_level.J):
                    continue
                component_key = (upper_term.beta, upper_level.J, lower_term.beta, lower_level.J)
                transition_registry.register_transition(
                    level_upper=level_registry.get_level(alpha=upper_term.term_id, J=upper_level.J),
                    level_lower=level_registry.get_level(alpha=lower_term.term_id, J=lower_level.J),
                    einstein_a_ul_sm1=_COMPONENT_A_UL_SM1[component_key],
                )

    return MultiLevelAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=multi_term_config.reference_lambda_A_air,
        atomic_mass_amu=multi_term_config.atomic_mass_amu,
    )
