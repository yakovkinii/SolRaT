import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    PrescribedRadiationStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import frequencies_around_line_sm1
from solrat.atom_model.shared.utility.log_setup import setup_logging


def build_case():
    r"""
    Shared LTE setup for atmosphere-interface equivalence tests.
    """
    model = PreconfiguredModels.multi_term_atom_mock_nofs_lte()
    transition = next(iter(model.config.transition_registry.transitions.values()))
    delta_v_turbulent_cm_sm1 = 1.5e5
    params = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=200.0,
        temperature_K=6000.0,
        delta_v_turbulent_cm_sm1=delta_v_turbulent_cm_sm1,
        macroscopic_velocity_cm_sm1=0.0,
        voigt_a=0.02,
    )
    nu = frequencies_around_line_sm1(
        transition.get_mean_transition_frequency_sm1(),
        params.delta_v_thermal_cm_sm1,
        half_width_doppler=3.0,
        step_doppler=0.5,
    )
    angles = Angles(chi=0.0, theta=0.0, gamma=0.0, chi_B=0.4, theta_B=0.6)
    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700.0)
    return model, params, delta_v_turbulent_cm_sm1, nu, angles, initial_stokes


def assert_same_stokes(test_case: unittest.TestCase, left: Stokes, right: Stokes):
    r"""
    Compare all four Stokes components.
    """
    test_case.assertTrue(np.array_equal(left.nu, right.nu))
    np.testing.assert_allclose(left.I, right.I, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(left.Q, right.Q, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(left.U, right.U, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(left.V, right.V, rtol=1e-12, atol=1e-20)


def line_center_opacity(model, nu, params, angles):
    r"""
    Line-center opacity with ``N = 1`` for matching a physical stratified column to a slab optical depth.
    """
    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    see.fill_all_equations(
        atmosphere_parameters=params,
        radiation_tensor_in_magnetic_frame=model.RadiationTensor(),
    )
    rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
    rte.N = 1.0
    rtc = rte.calculate_all_coefficients(
        atmosphere_parameters=params,
        angles=angles,
        rho=see.get_solution(),
    )
    return float(np.max(np.abs(rtc.get_eta_I())))


class TestAtmosphereEquivalence(unittest.TestCase):
    def test_identical_multislab_equals_single_slab_with_summed_optical_depths(self):
        r"""
        Two identical constant-property slabs in sequence reproduce one constant-property slab whose
        line and continuum optical depths are the sums of the two slab optical depths.
        """
        setup_logging()
        model, params, _delta_v_turbulent_cm_sm1, nu, angles, initial_stokes = build_case()
        radiation_tensor = model.RadiationTensor()

        half_line_tau = 0.35
        half_continuum_tau = 0.015
        two_slab = MultiSlabAtmosphere(
            ConstantPropertySlabAtmosphere(
                model=model,
                radiation_tensor=radiation_tensor,
                atmosphere_parameters=params,
                angles=angles,
                line_delta_tau=half_line_tau,
                continuum_delta_tau=half_continuum_tau,
            ),
            ConstantPropertySlabAtmosphere(
                model=model,
                radiation_tensor=radiation_tensor,
                atmosphere_parameters=params,
                angles=angles,
                line_delta_tau=half_line_tau,
                continuum_delta_tau=half_continuum_tau,
            ),
        )
        single_slab = ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            atmosphere_parameters=params,
            angles=angles,
            line_delta_tau=2.0 * half_line_tau,
            continuum_delta_tau=2.0 * half_continuum_tau,
        )

        assert_same_stokes(
            self,
            two_slab.forward(initial_stokes=initial_stokes),
            single_slab.forward(initial_stokes=initial_stokes),
        )

    def test_constant_prescribed_stratified_equals_constant_property_slab(self):
        r"""
        A constant two-node prescribed-JKQ stratified atmosphere reproduces a single
        constant-property slab when the physical height is chosen to give the same line and
        continuum optical depths.
        """
        setup_logging()
        model, params, delta_v_turbulent_cm_sm1, nu, angles, initial_stokes = build_case()
        radiation_tensor = model.RadiationTensor()

        line_tau = 0.7
        continuum_tau = 0.03
        eta_peak = line_center_opacity(model, nu, params, angles)
        height = line_tau / eta_peak
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=[0.0, height],
            temperature_K=params.temperature_K,
            number_density_cm3=1.0,
            magnetic_field_gauss=params.magnetic_field_gauss,
            theta_B=angles.theta_B,
            chi_B=angles.chi_B,
            delta_v_turbulent_cm_sm1=delta_v_turbulent_cm_sm1,
            voigt_a=params.voigt_a,
            continuum_opacity_cm_m1=continuum_tau / height,
        )
        stratified = PrescribedRadiationStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            radiation_tensor=radiation_tensor,
            los_theta=angles.theta,
            los_chi=angles.chi,
            los_gamma=angles.gamma,
            transfer_scheme="delo_constant",
        )
        slab = ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            atmosphere_parameters=params,
            angles=angles,
            line_delta_tau=line_tau,
            continuum_delta_tau=continuum_tau,
        )

        assert_same_stokes(
            self,
            stratified.forward(initial_stokes=initial_stokes),
            slab.forward(initial_stokes=initial_stokes),
        )


if __name__ == "__main__":
    unittest.main()
