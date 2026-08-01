import unittest

import numpy as np
from numpy import pi

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestRadiativeTransferEquationsOperatorCache(unittest.TestCase):
    r"""
    The compiled-operator cache is an opt-in memoization: enabling it must not change the
    :math:`\eta_A / \eta_S` coefficients (only avoid rebuilding), the cache must be keyed by
    (angles, atmosphere), and clearing or disabling it must invalidate it.
    """

    def _build(self):
        r"""Build a two-level atom with a solved rho, returning (model, nu, angles, atmosphere, rho)."""
        setup_logging()
        level_registry = LevelRegistry()
        level_registry.register_level(alpha="1s", J=0, energy_cmm1=200_000, g=1.0)
        level_registry.register_level(alpha="2p", J=1, energy_cmm1=220_000, g=1.0)  # 20000 cm^-1 line (~5000 A)
        transition_registry = TransitionRegistry()
        transition_registry.register_transition(
            level_upper=level_registry.get_level(alpha="2p", J=1),
            level_lower=level_registry.get_level(alpha="1s", J=0),
            einstein_a_ul_sm1=1e7,
        )
        model = Models.multi_level_atom().configure(
            config=Models.multi_level_atom().Config(
                level_registry=level_registry,
                transition_registry=transition_registry,
                atomic_mass_amu=4.0,
                reference_lambda_A_air=np.nan,
            )
        )
        transition = next(iter(model.config.transition_registry.transitions.values()))
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config, magnetic_field_gauss=0, temperature_K=7000
        )
        nu0 = transition.get_mean_transition_frequency_sm1()
        delta_nu_D = nu0 * atmosphere_parameters.delta_v_thermal_cm_sm1 / c_cm_sm1
        nu = np.arange(nu0 - 4 * delta_nu_D, nu0 + 4 * delta_nu_D, 0.5 * delta_nu_D)
        angles = Angles(chi=pi / 5, theta=pi / 7, gamma=pi / 9, chi_B=pi / 3, theta_B=pi / 5)

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
        assert not np.allclose(eta_a, eta_a_hotter)

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
