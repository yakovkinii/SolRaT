import pathlib

from solrat.atom_model.shared.common_api.nlte_state import NLTEState

STATE_DIR = pathlib.Path(__file__).resolve().parent


def state_path(demo_file: str, suffix: str = "") -> pathlib.Path:
    r"""
    Path of the warm-start state ``.npz`` for a demo, named after the demo file and kept in the
    ``state`` directory next to the demos.
    """
    stem = pathlib.Path(demo_file).stem
    return STATE_DIR / f"{stem}{suffix}.npz"


def load_warm_state(demo_file: str, warm_start: bool = True, suffix: str = ""):
    r"""
    Load the converged state saved for ``demo_file``, or return ``None`` when warm-start is disabled or
    no usable state is stored, so the caller cold-starts as usual. Pass ``__file__`` from the demo.

    :param demo_file: the calling demo's ``__file__``.
    :param warm_start: when False, always return ``None`` (a regular cold start).
    :param suffix: optional state-file suffix for demos with multiple parameter runs.
    :return: a :class:`NLTEState` to warm-start from, or ``None``.
    """
    if not warm_start:
        return None
    try:
        return NLTEState.load(str(state_path(demo_file, suffix=suffix)))
    except (OSError, ValueError, KeyError):
        return None


def save_warm_state(demo_file: str, atmosphere, suffix: str = "") -> None:
    r"""
    Save the converged state of ``atmosphere`` for the next warm start of ``demo_file``, always (even
    after a run that itself warm-started). Creates the ``state`` directory on first use.

    :param demo_file: the calling demo's ``__file__``.
    :param atmosphere: a solved :class:`NLTEStratifiedAtmosphere`.
    :param suffix: optional state-file suffix for demos with multiple parameter runs.
    """
    path = state_path(demo_file, suffix=suffix)
    atmosphere.get_state().save(str(path))
