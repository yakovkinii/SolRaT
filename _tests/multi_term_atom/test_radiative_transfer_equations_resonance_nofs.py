import logging
import unittest

import numpy as np
from yatools import logging_config

from src.multi_term_atom.atomic_data.mock import get_mock_atom_data
from src.multi_term_atom.legacy.radiative_transfer_equations_legacy import MultiTermAtomRTELegacy
from src.multi_term_atom.legacy.statistical_equilibrium_equations_legacy import MultiTermAtomSEELegacy
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        logging_config.init(logging.INFO)
        level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data(
            fine_structure=False
        )
        nu = np.arange(reference_nu_sm1 - 1e11, reference_nu_sm1 + 1e11, 1e9)  # Hz

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=5_000_00)
        radiation_tensor = (
            RadiationTensor(transition_registry=transition_registry)
            .fill_planck(T_K=5000)
            .rotate_to_magnetic_frame(
                chi_B=angles.chi_B,
                theta_B=angles.theta_B,
            )
        )

        see_legacy = MultiTermAtomSEELegacy(
            level_registry=level_registry,
            transition_registry=transition_registry,
            disable_r_s=True,
            disable_n=True,
        )
        see = MultiTermAtomSEE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            disable_r_s=True,
            disable_n=True,
        )

        see_legacy.add_all_equations(
            atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
        )
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
        )
        rho_legacy = see_legacy.get_solution_direct()
        rho = see.get_solution()

        rte_legacy = MultiTermAtomRTELegacy(
            transition_registry=transition_registry,
            nu=nu,
        )
        rte = MultiTermAtomRTE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            nu=nu,
        )

        eta_aI, eta_aQ, eta_aU, eta_aV = rte.eta_rho_a(
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )

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

        eta_sI, eta_sQ, eta_sU, eta_sV = rte.eta_rho_s(
            rho=rho,
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
        )

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
