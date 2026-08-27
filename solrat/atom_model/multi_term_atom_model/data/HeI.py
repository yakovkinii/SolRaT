from pathlib import Path

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.precomputed_data import PrecomputedData
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.engine.functions.decorators import log_function

_PRECOMPUTED_DIR = Path(__file__).parent / "HeI_precomputed"


@log_function
def get_He_I_D3_config() -> MultiTermAtomConfig:  # pragma: no cover
    r"""
    Config constructor for He I atom, with multiple transitions including D3.
    For details see A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433

    ``einstein_a_ul_sm1`` is the LL04 term coefficient
    :math:`A(\beta_u L_u S \to \beta_l L_l S)`. NIST usually tabulates fine-structure components
    :math:`A(\beta_u L_u S J_u \to \beta_l L_l S J_l)`. To convert those component values to the
    SolRaT multi-term coefficient, fix one upper :math:`J_u` and sum over all lower :math:`J_l`.
    The result should be independent of the chosen :math:`J_u` in exact LS coupling. In practice,
    average those per-:math:`J_u` sums over all upper :math:`J_u` values included in the NIST
    component list:

    .. math::
        A_\mathrm{SolRaT}
        =
        \frac{1}{N_{J_u}}
        \sum_{J_u}
        \sum_{J_l}
        A_\mathrm{NIST}(J_u \to J_l).

    Here :math:`N_{J_u}` is just the number of distinct upper fine-structure :math:`J_u` values
    included for the upper term; it is not a separate NIST quantity. Do not use the raw sum over all
    fine-structure component lines unless the upper term has only one :math:`J_u`.

    :return: :any:`MultiTermAtomConfig` instance
    """

    # Levels
    level_registry = LevelRegistry()
    level_registry.register_level(
        beta="2s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=159855.9726,
    )
    level_registry.register_level(
        beta="3s3",
        L=0,
        S=1,
        J=1,
        energy_cmm1=183236.7905,
    )
    level_registry.register_level(
        beta="2p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=169087.8291,
    )
    level_registry.register_level(
        beta="2p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=169086.8412,
    )
    level_registry.register_level(
        beta="2p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=169086.7647,
    )

    level_registry.register_level(
        beta="3p3",
        L=1,
        S=1,
        J=0,
        energy_cmm1=185564.8528,
    )
    level_registry.register_level(
        beta="3p3",
        L=1,
        S=1,
        J=1,
        energy_cmm1=185564.5817,
    )
    level_registry.register_level(
        beta="3p3",
        L=1,
        S=1,
        J=2,
        energy_cmm1=185564.5602,
    )
    level_registry.register_level(
        beta="3d3",
        L=2,
        S=1,
        J=1,
        energy_cmm1=186101.5908,
    )
    level_registry.register_level(
        beta="3d3",
        L=2,
        S=1,
        J=2,
        energy_cmm1=186101.5466,
    )
    level_registry.register_level(
        beta="3d3",
        L=2,
        S=1,
        J=3,
        energy_cmm1=186101.5440,
    )
    level_registry.validate()

    # Transitions
    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="2p3", L=1, S=1),
        term_lower=level_registry.get_term(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=1.022e7,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3p3", L=1, S=1),
        term_lower=level_registry.get_term(beta="2s3", L=0, S=1),
        einstein_a_ul_sm1=9.478e6,
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3s3", L=0, S=1),
        term_lower=level_registry.get_term(beta="2p3", L=1, S=1),
        # einstein_a_ul_sm1=3.080e6 + 9.259e6 + 1.540e7,  # upper term 3s3S has one J, so this sum is the term A
        einstein_a_ul_sm1=2.780e7,  # This one is from HAZEL
    )
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="3d3", L=2, S=1),
        term_lower=level_registry.get_term(beta="2p3", L=1, S=1),
        einstein_a_ul_sm1=7.060e7,
    )
    precomputed_data = PrecomputedData.load_from_directory(str(_PRECOMPUTED_DIR)) if _PRECOMPUTED_DIR.exists() else None
    return MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=5875.621,
        atomic_mass_amu=4.0,
        precomputed_data=precomputed_data,
    )
