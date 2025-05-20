import logging
import unittest

import numpy as np
from yatools import logging_config

from src.two_term_atom.atomic_data.mock import get_mock_atom_data
from src.two_term_atom.legacy.radiative_transfer_equations_legacy import TwoTermAtomRTELegacy
from src.two_term_atom.legacy.statistical_equilibrium_equations_legacy import TwoTermAtomSEELegacy
from src.two_term_atom.object.angles import Angles
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import TwoTermAtomRTE
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtomSEE


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        # (10.127)
        logging_config.init(logging.INFO)
        term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()
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

        see_legacy = TwoTermAtomSEELegacy(
            term_registry=term_registry,
            transition_registry=transition_registry,
            disable_r_s=True,
            disable_n=True,
        )
        see = TwoTermAtomSEE(
            term_registry=term_registry,
            transition_registry=transition_registry,
            disable_r_s=True,
            disable_n=True,
        )

        see_legacy.add_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )
        see.add_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )

        rho_legacy = see_legacy.get_solution_direct()
        rho = see.get_solution_direct()

        rte_legacy = TwoTermAtomRTELegacy(transition_registry=transition_registry, nu=nu)
        rte = TwoTermAtomRTE(term_registry=term_registry, transition_registry=transition_registry, nu=nu)

        eta_sI, eta_sQ, eta_sU, eta_sV = rte.eta_rho_s(
            rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles
        )

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
        epsilonI = rte.epsilon(eta_s=eta_sI, nu=nu)
        assert np.allclose(epsilonI_legacy, epsilonI, atol=1e-10, rtol=1e-10)
