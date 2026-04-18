import logging
import unittest

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.precomputed_data import PrecomputedData
from solrat.atom_model.shared.object.angles import Angles
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


class TestSEEPrecomputationReproducible(unittest.TestCase):
    """
    Verify that precomputed frames extracted from a fresh SEE can be injected into
    a new config and produce an identical rho solution.
    """

    def test_precomputation_roundtrip(self):
        logging_config.init(logging.INFO)

        model = PreconfiguredModels.multi_term_atom_mock()

        see_original = model.StatisticalEquilibriumEquations.from_model_config(model.config)

        precomputed = PrecomputedData(
            coherence_decay_df=see_original.coherence_decay_df,
            absorption_df=see_original.absorption_df,
            emission_df_e=see_original.emission_df_e,
            emission_df_s=see_original.emission_df_s,
            relaxation_df_a=see_original.relaxation_df_a,
            relaxation_df_e=see_original.relaxation_df_e,
            relaxation_df_s=see_original.relaxation_df_s,
        )

        config_cached = MultiTermAtomConfig(
            level_registry=model.config.level_registry,
            transition_registry=model.config.transition_registry,
            atomic_mass_amu=model.config.atomic_mass_amu,
            reference_lambda_A=model.config.reference_lambda_A,
            precomputed_data=precomputed,
        )

        see_cached = model.StatisticalEquilibriumEquations.from_model_config(config_cached)

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=100,
            temperature_K=7000,
        )
        radiation_tensor_mag = (
            model.RadiationTensor.from_model_config(model.config)
            .fill_NLTE_n_w_parametrized(h_arcsec=30)
            .rotate_to_magnetic_frame(angles=angles)
        )

        see_original.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )
        see_cached.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )

        rho = see_original.get_solution()
        rho_cached = see_cached.get_solution()

        for term in model.config.level_registry.terms.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                a = rho(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                b = rho_cached(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                assert np.allclose(a, b, rtol=1e-10, atol=1e-10), (
                    f"term={term.term_id} K={K} Q={Q} J={J} Jʹ={Jʹ}: "
                    f"precompute roundtrip mismatch {np.abs(a - b).max():.2e}"
                )
