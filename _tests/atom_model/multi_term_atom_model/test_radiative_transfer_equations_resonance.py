import logging
import unittest

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz


class TestRadiativeTransferEquationsResonance(unittest.TestCase):
    def test_radiative_transfer_equations_resonance(self):
        # (10.127)
        logging_config.init(logging.INFO)

        model = PreconfiguredModels.multi_term_atom_mock()
        reference_nu = lambda_A_to_frequency_hz(model.config.reference_lambda_A)

        nu = np.arange(reference_nu - 1e11, reference_nu + 1e11, 1e8)  # Hz

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_planck(T_K=5000)

        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=0,
            temperature_K=7000,
        )

        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )

        rho = see.get_solution()

        model_legacy = PreconfiguredModels.multi_term_atom_legacy_mock()
        # Compute SEE separately for legacy to test that SEE match
        see_legacy = model_legacy.StatisticalEquilibriumEquations.from_model_config(model_legacy.config)
        see_legacy.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )
        rho_legacy = see_legacy.get_solution()
        rte_legacy = model_legacy.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

        rtc = rte.calculate_all_coefficients(
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rtc_legacy = rte_legacy.calculate_all_coefficients(
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )

        assert np.allclose(rtc.get_eta_I(), rtc_legacy.get_eta_I(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_eta_Q(), rtc_legacy.get_eta_Q(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_eta_U(), rtc_legacy.get_eta_U(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_eta_V(), rtc_legacy.get_eta_V(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_rho_I(), rtc_legacy.get_rho_I(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_rho_Q(), rtc_legacy.get_rho_Q(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_rho_U(), rtc_legacy.get_rho_U(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_rho_V(), rtc_legacy.get_rho_V(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_epsilon_I(), rtc_legacy.get_epsilon_I(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_epsilon_Q(), rtc_legacy.get_epsilon_Q(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_epsilon_U(), rtc_legacy.get_epsilon_U(), atol=1e-10, rtol=1e-10)
        assert np.allclose(rtc.get_epsilon_V(), rtc_legacy.get_epsilon_V(), atol=1e-10, rtol=1e-10)

        eta_a = rte.calculate_eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        eta_aI, eta_aQ, eta_aU, eta_aV = eta_a[0], eta_a[1], eta_a[2], eta_a[3]

        eta_aI_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_aI_legacy_real = rte_legacy.eta_a(
            rho=rho_legacy, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_aI_legacy_imag = rte_legacy.rho_a(
            rho=rho_legacy, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )

        scale = np.max(np.abs(eta_aI_legacy))
        assert np.allclose(eta_aI / scale, eta_aI_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_aI) / scale, eta_aI_legacy_real / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_aI) / scale, eta_aI_legacy_imag / scale, atol=1e-10, rtol=1e-10)

        eta_aQ_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy, stokes_component_index=1, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        assert np.allclose(eta_aQ / scale, eta_aQ_legacy / scale, atol=1e-10, rtol=1e-10)

        eta_aU_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy, stokes_component_index=2, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        assert np.allclose(eta_aU / scale, eta_aU_legacy / scale, atol=1e-10, rtol=1e-10)

        eta_aV_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy, stokes_component_index=3, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        assert np.allclose(eta_aV / scale, eta_aV_legacy / scale, atol=1e-10, rtol=1e-10)

        eta_s = rte.calculate_eta_rho_s(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        eta_sI, eta_sQ, eta_sU, eta_sV = eta_s[0], eta_s[1], eta_s[2], eta_s[3]

        eta_sI_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_sI_legacy_real = rte_legacy.eta_s(
            rho=rho_legacy, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_sI_legacy_imag = rte_legacy.rho_s(
            rho=rho_legacy, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_sI_analytic = rte_legacy.eta_s_no_field(
            rho=rho, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        scale = np.max(np.abs(eta_sI_analytic))
        assert np.allclose(eta_sI / scale, eta_sI_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sI) / scale, eta_sI_legacy_real / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_sI) / scale, eta_sI_legacy_imag / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sI) / scale, eta_sI_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_sQ_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy, stokes_component_index=1, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_sQ_analytic = rte_legacy.eta_s_no_field(
            rho=rho, stokes_component_index=1, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        assert np.allclose(eta_sQ / scale, eta_sQ_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sQ) / scale, eta_sQ_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_sU_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy, stokes_component_index=2, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_sU_analytic = rte_legacy.eta_s_no_field(
            rho=rho, stokes_component_index=2, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        assert np.allclose(eta_sU / scale, eta_sU_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sU) / scale, eta_sU_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_sV_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy, stokes_component_index=3, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_sV_analytic = rte_legacy.eta_s_no_field(
            rho=rho, stokes_component_index=3, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        assert np.allclose(eta_sV / scale, eta_sV_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sV) / scale, eta_sV_analytic / scale, atol=1e-10, rtol=1e-10)

        epsilonI_legacy = rte_legacy.epsilon(eta_s=eta_sI_legacy, nu=nu)
        epsilonI = rte.calculate_epsilon(eta_s=eta_sI, nu=nu)
        assert np.allclose(epsilonI_legacy, epsilonI, atol=1e-10, rtol=1e-10)
