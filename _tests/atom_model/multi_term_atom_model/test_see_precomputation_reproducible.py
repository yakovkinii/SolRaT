import unittest

import numpy as np

from solrat.atom_model.model_registry import Models, PreconfiguredModels
from solrat.atom_model.multi_term_atom_model.data.HeI import _PRECOMPUTED_DIR, get_He_I_D3_config
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


class TestSEEPrecomputationReproducible(unittest.TestCase):
    """
    Verify that precomputed frames loaded from disk produce an identical rho solution
    to a fresh end-to-end computation.
    """

    def test_disk_precomputed_matches_fresh(self):
        """He I D3: loading CSVs from disk must give the same rho as computing from scratch."""
        setup_logging()

        # --- fresh model (no precomputed data) ---
        config_fresh = get_He_I_D3_config()
        config_fresh = MultiTermAtomConfig(
            level_registry=config_fresh.level_registry,
            transition_registry=config_fresh.transition_registry,
            atomic_mass_amu=config_fresh.atomic_mass_amu,
            reference_lambda_A_air=config_fresh.reference_lambda_A_air,
            precomputed_data=None,
        )
        model_fresh = Models.multi_term_atom().configure(config=config_fresh)

        # --- disk-precomputed model (CSVs loaded by get_He_I_D3_config) ---
        model_disk = PreconfiguredModels.multi_term_atom_HeID3()
        assert model_disk.config.precomputed_data is not None, (
            "He I D3 config should load precomputed data from disk. "
            f"Check that {_PRECOMPUTED_DIR} contains the expected CSV files."
        )

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        def _make_inputs(model):
            atm = model.AtmosphereParameters(
                model_config=model.config,
                magnetic_field_gauss=100,
                temperature_K=7000,
            )
            rad = (
                model.RadiationTensor.from_model_config(model.config)
                .fill_NLTE_n_w_allen(h_arcsec=30)
                .rotate_to_magnetic_frame(angles=angles)
            )
            return atm, rad

        atm_fresh, rad_fresh = _make_inputs(model_fresh)
        atm_disk, rad_disk = _make_inputs(model_disk)

        see_fresh = model_fresh.StatisticalEquilibriumEquations.from_model_config(model_fresh.config)
        see_fresh.fill_all_equations(
            atmosphere_parameters=atm_fresh,
            radiation_tensor_in_magnetic_frame=rad_fresh,
        )
        rho_fresh = see_fresh.get_solution()

        see_disk = model_disk.StatisticalEquilibriumEquations.from_model_config(model_disk.config)
        see_disk.fill_all_equations(
            atmosphere_parameters=atm_disk,
            radiation_tensor_in_magnetic_frame=rad_disk,
        )
        rho_disk = see_disk.get_solution()

        for term in model_fresh.config.level_registry.terms.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                a = rho_fresh(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                b = rho_disk(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                assert np.allclose(a, b, rtol=1e-10, atol=1e-10), (
                    f"term={term.term_id} K={K} Q={Q} J={J} Jʹ={Jʹ}: "
                    f"disk-precomputed mismatch {np.abs(a - b).max():.2e}"
                )
