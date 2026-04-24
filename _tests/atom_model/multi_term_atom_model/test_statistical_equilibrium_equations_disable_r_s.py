import logging
import unittest

import numpy as np

from solrat.atom_model.model_registry import Models, PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


class TestStatisticalEquilibriumEquationsDisableRs(unittest.TestCase):
    """
    Verify the disable_r_s flag: current and legacy SEE must agree for both
    settings, and the two settings must produce physically different solutions.
    """

    def _build_model_pair(self, base_model, base_legacy_model, disable_r_s: bool):
        """Clone the mock atom config with the requested disable_r_s setting."""
        cfg = base_model.config

        model = Models.multi_term_atom().configure(
            config=base_model.Config(
                level_registry=cfg.level_registry,
                transition_registry=cfg.transition_registry,
                atomic_mass_amu=cfg.atomic_mass_amu,
                reference_lambda_A_air=cfg.reference_lambda_A_air,
                disable_r_s=disable_r_s,
            )
        )
        model_legacy = Models.multi_term_atom_legacy().configure(
            config=base_legacy_model.Config(
                level_registry=cfg.level_registry,
                transition_registry=cfg.transition_registry,
                atomic_mass_amu=cfg.atomic_mass_amu,
                reference_lambda_A_air=cfg.reference_lambda_A_air,
                disable_r_s=disable_r_s,
            )
        )
        return model, model_legacy

    def _solve_both(self, model, model_legacy, atmosphere_parameters, radiation_tensor_mag):
        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see_legacy = model_legacy.StatisticalEquilibriumEquations.from_model_config(model_legacy.config)

        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )
        see_legacy.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )
        return see.get_solution(), see_legacy.get_solution()

    def test_disable_r_s(self):
        setup_logging(logging.INFO)

        base_model = PreconfiguredModels.multi_term_atom_mock()
        base_legacy = PreconfiguredModels.multi_term_atom_legacy_mock()

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )
        radiation_tensor = base_model.RadiationTensor.from_model_config(base_model.config).fill_NLTE_n_w_parametrized(
            h_arcsec=30
        )
        radiation_tensor_mag = radiation_tensor.rotate_to_magnetic_frame(angles=angles)

        atmosphere_parameters = base_model.AtmosphereParameters(
            model_config=base_model.config,
            magnetic_field_gauss=0,
            temperature_K=7000,
        )

        # --- Case 1: stimulated emission enabled ---
        model_on, model_legacy_on = self._build_model_pair(base_model, base_legacy, disable_r_s=False)
        rho_on, rho_legacy_on = self._solve_both(model_on, model_legacy_on, atmosphere_parameters, radiation_tensor_mag)

        # --- Case 2: stimulated emission disabled ---
        model_off, model_legacy_off = self._build_model_pair(base_model, base_legacy, disable_r_s=True)
        rho_off, rho_legacy_off = self._solve_both(
            model_off, model_legacy_off, atmosphere_parameters, radiation_tensor_mag
        )

        terms = base_model.config.level_registry.terms.values()

        # Current and legacy must agree for each setting
        for term in terms:
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                kwargs = dict(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                assert np.allclose(
                    rho_on(**kwargs), rho_legacy_on(**kwargs), rtol=1e-10, atol=1e-10
                ), "current vs legacy mismatch with disable_r_s=False"
                assert np.allclose(
                    rho_off(**kwargs), rho_legacy_off(**kwargs), rtol=1e-10, atol=1e-10
                ), "current vs legacy mismatch with disable_r_s=True"

        # The two settings must produce a physically different result
        max_diff = 0.0
        for term in terms:
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                kwargs = dict(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                diff = np.abs(rho_on(**kwargs) - rho_off(**kwargs)).max()
                if diff > max_diff:
                    max_diff = diff

        assert max_diff > 1e-6, (
            f"disable_r_s=True/False produced nearly identical solutions (max diff {max_diff:.2e}); "
            "stimulated emission appears to have no effect"
        )
