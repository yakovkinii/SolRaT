import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestMultiTermRadiativeTransferEquationsOperatorCache(unittest.TestCase):
    r"""
    The compiled-operator cache is an opt-in memoization: enabling it must not change the
    :math:`\eta_A / \eta_S` coefficients (only avoid rebuilding), the cache must be keyed by
    (angles, atmosphere), and clearing or disabling it must invalidate it.
    """

    def _build(self):
        r"""Build the mock multi-term atom with a solved rho; return (model, nu, angles, atmosphere, rho)."""
        setup_logging()
        model = PreconfiguredModels.multi_term_atom_mock()
        reference_lambda_A_air = model.config.reference_lambda_A_air
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.5,
            upper_wavelength_A=reference_lambda_A_air + 0.5,
            step_A=2e-3,
        )
        angles = Angles(chi=np.pi / 5, theta=np.pi / 7, gamma=np.pi / 9, chi_B=np.pi / 3, theta_B=np.pi / 5)
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config, magnetic_field_gauss=0, temperature_K=7000
        )
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(h_arcsec=30)
        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )
        rho = see.get_solution()
        return model, nu, angles, atmosphere_parameters, rho

    def test_cache_on_off_give_identical_coefficients(self):
        model, nu, angles, atmosphere_parameters, rho = self._build()
        arguments = dict(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)

        rte_off = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rte_on = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rte_on.use_operator_cache = True

        eta_a_off = rte_off.calculate_eta_rho_a(**arguments)
        eta_s_off = rte_off.calculate_eta_rho_s(**arguments)
        eta_a_on = rte_on.calculate_eta_rho_a(**arguments)
        eta_s_on = rte_on.calculate_eta_rho_s(**arguments)
        eta_a_on_again = rte_on.calculate_eta_rho_a(**arguments)  # served from the cache

        assert np.array_equal(eta_a_off, eta_a_on)
        assert np.array_equal(eta_s_off, eta_s_on)
        assert np.array_equal(eta_a_on, eta_a_on_again)

    def test_cache_is_opt_in(self):
        model, nu, angles, atmosphere_parameters, rho = self._build()
        arguments = dict(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)

        rte_off = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rte_off.calculate_eta_rho_a(**arguments)
        rte_off.calculate_eta_rho_s(**arguments)
        assert rte_off.eta_rho_a_cache.get(angles, atmosphere_parameters) is None
        assert rte_off.eta_rho_s_cache.get(angles, atmosphere_parameters) is None

        rte_on = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rte_on.use_operator_cache = True
        rte_on.calculate_eta_rho_a(**arguments)
        rte_on.calculate_eta_rho_s(**arguments)
        assert rte_on.eta_rho_a_cache.get(angles, atmosphere_parameters) is not None
        assert rte_on.eta_rho_s_cache.get(angles, atmosphere_parameters) is not None

    def test_cache_key_distinguishes_atmosphere(self):
        model, nu, angles, atmosphere_parameters, rho = self._build()
        hotter_atmosphere = model.AtmosphereParameters(
            model_config=model.config, magnetic_field_gauss=0, temperature_K=14000
        )
        rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rte.use_operator_cache = True

        eta_a = rte.calculate_eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        eta_a_hotter = rte.calculate_eta_rho_a(rho=rho, atmosphere_parameters=hotter_atmosphere, angles=angles)
        # A different atmosphere is a different key, so a different (broader-profile) operator is built.
        scale = np.max(np.abs(eta_a))
        assert not np.allclose(eta_a / scale, eta_a_hotter / scale, atol=1e-6)

    def test_clear_invalidates(self):
        model, nu, angles, atmosphere_parameters, rho = self._build()
        rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
        rte.use_operator_cache = True

        rte.calculate_eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        assert rte.eta_rho_a_cache.get(angles, atmosphere_parameters) is not None
        rte.clear_operator_cache()
        assert rte.eta_rho_a_cache.get(angles, atmosphere_parameters) is None


if __name__ == "__main__":
    unittest.main()
