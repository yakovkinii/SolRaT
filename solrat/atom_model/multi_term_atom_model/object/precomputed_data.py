import os
from dataclasses import dataclass, field
from typing import Union

import pandas as pd

from solrat.engine.generators.merge_frame import Frame, FrameFactor


def _frame_to_csv(frame: Frame, path: str) -> None:  # pragma: no cover
    """Save a Frame to CSV by evaluating all unmerged factors into a single 'coefficient' column."""
    f = frame.copy()
    # Merge any unmerged factors
    for name in list(f.factors.keys()):
        if not f.factors[name].merged:
            f.merge_factor(name)
    # Combine multiple merged factors into one column
    if len(f.factors) > 1:
        combined_name = f.combine_all_merged_factors()
    else:
        combined_name = next(iter(f.factors)) if f.factors else None
    # Rename to 'coefficient' if needed
    if combined_name is not None and combined_name != "coefficient":
        f.frame = f.frame.rename(columns={combined_name: "coefficient"})
    f.frame.to_csv(path, index=False)


def _frame_from_csv(path: str) -> Frame:
    """Load a Frame from CSV. The 'coefficient' column is registered as a merged factor."""
    df = pd.read_csv(path)
    frame = Frame(base_frame=df)
    loop_cols = [c for c in df.columns if c != "coefficient"]
    frame.factors["coefficient"] = FrameFactor(
        name="coefficient",
        dependencies=loop_cols,
        merged=True,
    )
    return frame


_FILE_MAP = {
    "coherence_decay_frame": "coherence_decay_n0.csv",
    "coherence_decay_frame_n_1": "coherence_decay_n1.csv",
    "absorption_frame": "absorption.csv",
    "emission_e_frame": "emission_e.csv",
    "emission_s_frame": "emission_s.csv",
    "relaxation_e_frame": "relaxation_e.csv",
    "relaxation_a_frame": "relaxation_a.csv",
    "relaxation_s_frame": "relaxation_s.csv",
}


@dataclass
class PrecomputedData:
    """
    Container for precomputed atom-specific SEE frames.

    Frames that do not depend on the radiation tensor or atmosphere parameters can be
    precomputed once, saved to CSV via :meth:`save_to_directory`, and reloaded via
    :meth:`load_from_directory`.  Frames that still require the radiation tensor (absorption,
    emission_s, relaxation_a, relaxation_s) are stored with the atom-specific factor already
    evaluated; the radiation-tensor factor is applied per :meth:`fill_all_equations` call.
    """

    coherence_decay_frame: Frame
    absorption_frame: Frame
    emission_e_frame: Frame
    emission_s_frame: Frame
    relaxation_e_frame: Frame
    relaxation_a_frame: Frame
    relaxation_s_frame: Frame
    coherence_decay_frame_n_1: Union[Frame, None] = field(default=None)

    def save_to_directory(self, directory: str) -> None:  # pragma: no cover
        """Save all frames to CSV files in *directory*."""
        os.makedirs(directory, exist_ok=True)
        for attr, filename in _FILE_MAP.items():
            frame = getattr(self, attr, None)
            if frame is not None:
                _frame_to_csv(frame, os.path.join(directory, filename))

    @classmethod
    def load_from_directory(cls, directory: str) -> "PrecomputedData":
        """Load all frames from CSV files in *directory*."""

        def _load(filename: str) -> Frame:
            return _frame_from_csv(os.path.join(directory, filename))

        n1_path = os.path.join(directory, _FILE_MAP["coherence_decay_frame_n_1"])
        return cls(
            coherence_decay_frame=_load(_FILE_MAP["coherence_decay_frame"]),
            absorption_frame=_load(_FILE_MAP["absorption_frame"]),
            emission_e_frame=_load(_FILE_MAP["emission_e_frame"]),
            emission_s_frame=_load(_FILE_MAP["emission_s_frame"]),
            relaxation_e_frame=_load(_FILE_MAP["relaxation_e_frame"]),
            relaxation_a_frame=_load(_FILE_MAP["relaxation_a_frame"]),
            relaxation_s_frame=_load(_FILE_MAP["relaxation_s_frame"]),
            coherence_decay_frame_n_1=(
                _load(_FILE_MAP["coherence_decay_frame_n_1"]) if os.path.exists(n1_path) else None
            ),
        )
