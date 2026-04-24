import unittest

import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestRadiativeTransferEquationsResonanceNoFS(unittest.TestCase):
    def test_radiative_transfer_equations_resonance_nofs(self):
        # (10.127)
        setup_logging()

        model = PreconfiguredModels.multi_term_atom_mock_nofs()
        reference_lambda_A_air = model.config.reference_lambda_A_air
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 1,
            upper_wavelength_A=reference_lambda_A_air + 1,
            step_A=5e-4,
        )

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_planck(temperature_K=5000)

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

        model_legacy = PreconfiguredModels.multi_term_atom_legacy_mock_nofs()
        # Compute SEE separately for legacy to test that SEE match
        see_legacy = model_legacy.StatisticalEquilibriumEquations.from_model_config(model_legacy.config)
        see_legacy.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )
        rho_legacy = see_legacy.get_solution()
        rte_legacy = model_legacy.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

        eta_a = rte.calculate_eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        eta_aI, eta_aQ, eta_aU, eta_aV = eta_a[0], eta_a[1], eta_a[2], eta_a[3]
        eta_aI_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy,
            stokes_component_index=0,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_aI_analytic = rte_legacy.eta_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=0,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_aI_analytic = rte_legacy.rho_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=0,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        scale = np.max(np.abs(eta_aI_analytic))
        assert np.allclose(eta_aI / scale, eta_aI_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_aI) / scale, eta_aI_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_aI) / scale, rho_aI_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_aQ_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy,
            stokes_component_index=1,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_aQ_analytic = rte_legacy.eta_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=1,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_aQ_analytic = rte_legacy.rho_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=1,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        assert np.allclose(eta_aQ / scale, eta_aQ_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_aQ) / scale, eta_aQ_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_aQ) / scale, rho_aQ_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_aU_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy,
            stokes_component_index=2,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_aU_analytic = rte_legacy.eta_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=2,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_aU_analytic = rte_legacy.rho_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=2,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        assert np.allclose(eta_aU / scale, eta_aU_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_aU) / scale, eta_aU_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_aU) / scale, rho_aU_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_aV_legacy = rte_legacy.eta_rho_a(
            rho=rho_legacy,
            stokes_component_index=3,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_aV_analytic = rte_legacy.eta_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=3,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_aV_analytic = rte_legacy.rho_a_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=3,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        assert np.allclose(eta_aV / scale, eta_aV_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_aV) / scale, eta_aV_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_aV) / scale, rho_aV_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_s = rte.calculate_eta_rho_s(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        eta_sI, eta_sQ, eta_sU, eta_sV = eta_s[0], eta_s[1], eta_s[2], eta_s[3]

        eta_sI_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy,
            stokes_component_index=0,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_sI_analytic = rte_legacy.eta_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=0,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_sI_analytic = rte_legacy.rho_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=0,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        scale = np.max(np.abs(eta_sI_analytic))
        assert np.allclose(eta_sI / scale, eta_sI_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sI) / scale, eta_sI_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_sI) / scale, rho_sI_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_sQ_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy,
            stokes_component_index=1,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_sQ_analytic = rte_legacy.eta_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=1,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_sQ_analytic = rte_legacy.rho_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=1,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        assert np.allclose(eta_sQ / scale, eta_sQ_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sQ) / scale, eta_sQ_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_sQ) / scale, rho_sQ_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_sU_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy,
            stokes_component_index=2,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_sU_analytic = rte_legacy.eta_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=2,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_sU_analytic = rte_legacy.rho_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=2,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        assert np.allclose(eta_sU / scale, eta_sU_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sU) / scale, eta_sU_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_sU) / scale, rho_sU_analytic / scale, atol=1e-10, rtol=1e-10)

        eta_sV_legacy = rte_legacy.eta_rho_s(
            rho=rho_legacy,
            stokes_component_index=3,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        eta_sV_analytic = rte_legacy.eta_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=3,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        rho_sV_analytic = rte_legacy.rho_s_no_field_no_fine_structure(
            rho=rho,
            stokes_component_index=3,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )
        assert np.allclose(eta_sV / scale, eta_sV_legacy / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.real(eta_sV) / scale, eta_sV_analytic / scale, atol=1e-10, rtol=1e-10)
        assert np.allclose(np.imag(eta_sV) / scale, rho_sV_analytic / scale, atol=1e-10, rtol=1e-10)
