import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.milne_eddington_slab import MilneEddingtonSlabAtmosphere
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    PrescribedRadiationStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import frequencies_around_line_sm1
from solrat.engine.generators.compiled_operator import CompiledOperator


def build_case():
    r"""
    Shared mock-atom setup for atmosphere cache-policy tests.
    """
    model = PreconfiguredModels.multi_term_atom_mock_nofs_lte()
    transition = next(iter(model.config.transition_registry.transitions.values()))
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=100.0,
        temperature_K=6000.0,
        delta_v_turbulent_cm_sm1=1.0e5,
        macroscopic_velocity_cm_sm1=0.0,
        voigt_a=0.01,
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(),
        atmosphere_parameters.delta_v_thermal_cm_sm1,
        half_width_doppler=3.0,
        step_doppler=0.5,
    )
    angles = Angles(chi=0.0, theta=0.2, gamma=0.0, chi_B=0.4, theta_B=0.6)
    return model, atmosphere_parameters, nu, angles


def see_rte(model, nu):
    r"""
    Real SEE/RTE objects for the configured mock atom.
    """
    return (
        model.StatisticalEquilibriumEquations.from_model_config(model.config),
        model.RadiativeTransferEquations.from_model_config(model.config, nu=nu),
    )


def slab_like(cls, model, atmosphere_parameters, angles, see=None, rte=None):
    common = dict(
        model=model,
        radiation_tensor=model.RadiationTensor(),
        atmosphere_parameters=atmosphere_parameters,
        angles=angles,
        see=see,
        rte=rte,
    )
    if cls is ConstantPropertySlabAtmosphere:
        return cls(line_delta_tau=1.0, continuum_delta_tau=1.0, **common)
    return cls(line_to_continuum_ratio=1.0, source_gradient=1.0, **common)


def stratification_for(model, atmosphere_parameters, angles):
    return StratifiedAtmosphere(
        model=model,
        height_cm=[0.0, 1.0],
        temperature_K=atmosphere_parameters.temperature_K,
        number_density_cm3=1.0,
        magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
        theta_B=angles.theta_B,
        chi_B=angles.chi_B,
        delta_v_turbulent_cm_sm1=atmosphere_parameters.delta_v_turbulent_cm_sm1,
        voigt_a=atmosphere_parameters.voigt_a,
        continuum_opacity_cm_m1=0.0,
    )


def cached_operator():
    return CompiledOperator(keys=[()], coefficients=np.array([1.0 + 0.0j]))


class TestAtmosphereCachePolicy(unittest.TestCase):
    def test_constant_and_milne_use_supplied_rte_atom_cache_only(self):
        r"""
        Slab atmospheres may reuse a caller-supplied RTE object, but keep the RTE operator cache off.
        """
        model, atmosphere_parameters, nu, angles = build_case()
        for atmosphere_cls in (ConstantPropertySlabAtmosphere, MilneEddingtonSlabAtmosphere):
            see, rte = see_rte(model, nu)
            rte.use_operator_cache = True
            rte.eta_rho_a_cache.add("sentinel", operator=cached_operator())
            atmosphere = slab_like(atmosphere_cls, model, atmosphere_parameters, angles, see=see, rte=rte)

            assert atmosphere._get_rte(nu) is rte
            assert rte.use_operator_cache is False
            rte.eta_rho_a_cache.enabled = True
            assert rte.eta_rho_a_cache.get("sentinel") is not None

    def test_constant_and_milne_construct_fresh_rte_without_supplied_rte(self):
        r"""
        Without a caller-supplied RTE, slab atmospheres keep the previous per-forward construction
        behavior instead of silently binding themselves to the first frequency grid.
        """
        model, atmosphere_parameters, nu, angles = build_case()
        for atmosphere_cls in (ConstantPropertySlabAtmosphere, MilneEddingtonSlabAtmosphere):
            atmosphere = slab_like(atmosphere_cls, model, atmosphere_parameters, angles)

            first = atmosphere._get_rte(nu)
            second = atmosphere._get_rte(nu)

            assert first is not second

    def test_supplied_rte_frequency_grid_must_match(self):
        r"""
        A reusable RTE carries atom-level data for a specific frequency grid.
        """
        model, atmosphere_parameters, nu, angles = build_case()
        see, rte = see_rte(model, nu)
        atmosphere = slab_like(ConstantPropertySlabAtmosphere, model, atmosphere_parameters, angles, see=see, rte=rte)

        with self.assertRaises(AssertionError):
            atmosphere._get_rte(nu[:-1])

    def test_stratified_reuses_supplied_see_and_rte(self):
        r"""
        Stratified atmospheres can be given SEE/RTE objects whose atom-level caches live outside a
        single atmosphere call.
        """
        model, atmosphere_parameters, nu, angles = build_case()
        see, rte = see_rte(model, nu)
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification_for(model, atmosphere_parameters, angles),
            los_theta=angles.theta,
            n_mu_quadrature=2,
            n_phi_quadrature=3,
            see=see,
            rte=rte,
        )

        assert atmosphere._get_see() is see
        assert atmosphere._get_rte(nu) is rte

    def test_self_consistent_operator_cache_is_cleared_and_restored(self):
        r"""
        The self-consistent stratified atmosphere owns the operator cache during the iteration and
        invalidates it when the atmosphere call is done.
        """
        model, _atmosphere_parameters, nu, _angles = build_case()
        _see, rte = see_rte(model, nu)
        rte.use_operator_cache = False
        rte.eta_rho_a_cache.enabled = True
        rte.eta_rho_a_cache.add("sentinel", operator=cached_operator())
        rte.eta_rho_a_cache.enabled = False

        previous = NLTEStratifiedAtmosphere._set_operator_cache_enabled(rte, True)
        assert previous is False
        assert rte.use_operator_cache is True

        NLTEStratifiedAtmosphere._clear_operator_cache(rte)
        assert rte.eta_rho_a_cache.get("sentinel") is None

        NLTEStratifiedAtmosphere._restore_operator_cache_enabled(rte, previous)
        assert rte.use_operator_cache is False

    def test_prescribed_stratified_policy_disables_operator_cache_without_clearing(self):
        r"""
        Prescribed-radiation stratified transfer may reuse atom-level caches, but it does not own an
        operator-cache lifetime.
        """
        model, _atmosphere_parameters, nu, _angles = build_case()
        _see, rte = see_rte(model, nu)
        rte.use_operator_cache = True
        rte.eta_rho_a_cache.add("sentinel", operator=cached_operator())

        PrescribedRadiationStratifiedAtmosphere._set_operator_cache_enabled(rte, False)

        assert rte.use_operator_cache is False
        rte.eta_rho_a_cache.enabled = True
        assert rte.eta_rho_a_cache.get("sentinel") is not None


if __name__ == "__main__":
    unittest.main()
