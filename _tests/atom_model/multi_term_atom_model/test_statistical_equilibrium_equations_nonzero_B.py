import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.looping import PROJECTION, TRIANGULAR
from solrat.engine.generators.nested_loops import nested_loops


class TestStatisticalEquilibriumEquationsNonzeroB(unittest.TestCase):
    """
    Verify that the current and legacy SEE implementations agree at non-zero
    magnetic field, exercising the Larmor precession path in the coherence-decay
    kernel.
    """

    def _run_for_model_pair(self, model, model_legacy, magnetic_field_gauss, angles, radiation_tensor):
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=magnetic_field_gauss,
            temperature_K=7000,
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see_legacy = model_legacy.StatisticalEquilibriumEquations.from_model_config(model_legacy.config)

        radiation_tensor_mag = radiation_tensor.rotate_to_magnetic_frame(angles=angles)

        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )
        see_legacy.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )

        rho = see.get_solution()
        rho_legacy = see_legacy.get_solution()

        for term in model.config.level_registry.terms.values():
            for J, Jʹ, K, Q in nested_loops(
                J=TRIANGULAR(term.L, term.S),
                Jʹ=TRIANGULAR(term.L, term.S),
                K=TRIANGULAR("J", "Jʹ"),
                Q=PROJECTION("K"),
            ):
                a = rho(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                b = rho_legacy(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jʹ)
                assert np.allclose(a, b, rtol=1e-10, atol=1e-8), (
                    f"term={term.term_id} K={K} Q={Q} J={J} Jʹ={Jʹ}: "
                    f"max diff {np.abs(a - b).max():.2e} exceeds tolerance"
                )

    def test_nonzero_B_mock(self):
        """Mock atom (with fine structure), B = 100 G, anisotropic radiation."""
        setup_logging()

        model = PreconfiguredModels.multi_term_atom_mock()
        model_legacy = PreconfiguredModels.multi_term_atom_legacy_mock()

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=30)

        self._run_for_model_pair(
            model=model,
            model_legacy=model_legacy,
            magnetic_field_gauss=100,
            angles=angles,
            radiation_tensor=radiation_tensor,
        )

    def test_nonzero_B_mock_nofs(self):
        """Mock atom (no fine structure), B = 50 G, anisotropic radiation."""
        setup_logging()

        model = PreconfiguredModels.multi_term_atom_mock_nofs()
        model_legacy = PreconfiguredModels.multi_term_atom_legacy_mock_nofs()

        angles = Angles(
            chi=np.pi / 6,
            theta=np.pi / 4,
            gamma=0,
            chi_B=0,
            theta_B=np.pi / 3,
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=30)

        self._run_for_model_pair(
            model=model,
            model_legacy=model_legacy,
            magnetic_field_gauss=50,
            angles=angles,
            radiation_tensor=radiation_tensor,
        )
