from dataclasses import dataclass

import pandas as pd


@dataclass
class PrecomputedData:
    """
    Container class for the precomputed atomic data.
    """

    coherence_decay_df: pd.DataFrame
    absorption_df: pd.DataFrame
    emission_df_e: pd.DataFrame
    emission_df_s: pd.DataFrame
    relaxation_df_a: pd.DataFrame
    relaxation_df_e: pd.DataFrame
    relaxation_df_s: pd.DataFrame
