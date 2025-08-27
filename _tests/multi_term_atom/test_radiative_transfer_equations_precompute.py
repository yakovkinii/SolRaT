import logging
import unittest

import numpy as np
from yatools import logging_config

from src.multi_term_atom.atomic_data.mock import get_mock_atom_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        # (10.127)
        logging_config.init(logging.INFO)
        level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()
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

        see = MultiTermAtomSEE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            disable_r_s=True,
            disable_n=True,
        )

        see.add_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor,
        )

        rho = see.get_solution_direct()

        rte = MultiTermAtomRTE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            nu=nu,
        )

        rte_angles = MultiTermAtomRTE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            nu=nu,
            angles=angles,
        )

        rte_angles_magnetic = MultiTermAtomRTE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            nu=nu,
            angles=angles,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
        )

        rte_angles_magnetic_rho = MultiTermAtomRTE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            nu=nu,
            angles=angles,
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            rho=rho,
        )

        eta_a_all = rte.eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
        eta_s_all = rte.eta_rho_s(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)

        eta_a_all_angles = rte_angles.eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters)
        eta_s_all_angles = rte_angles.eta_rho_s(rho=rho, atmosphere_parameters=atmosphere_parameters)

        eta_a_all_angles_magnetic = rte_angles_magnetic.eta_rho_a(rho=rho, atmosphere_parameters=atmosphere_parameters)
        eta_s_all_angles_magnetic = rte_angles_magnetic.eta_rho_s(rho=rho, atmosphere_parameters=atmosphere_parameters)

        eta_a_all_angles_magnetic_rho = rte_angles_magnetic_rho.eta_rho_a(atmosphere_parameters=atmosphere_parameters)
        eta_s_all_angles_magnetic_rho = rte_angles_magnetic_rho.eta_rho_s(atmosphere_parameters=atmosphere_parameters)

        scale = np.max(np.abs(eta_a_all[0]))

        for stokes_component_index in range(4):
            assert np.allclose(
                eta_a_all[stokes_component_index] / scale,
                eta_a_all_angles[stokes_component_index] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
            assert np.allclose(
                eta_s_all[stokes_component_index] / scale,
                eta_s_all_angles[stokes_component_index] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
            assert np.allclose(
                eta_a_all[stokes_component_index] / scale,
                eta_a_all_angles_magnetic[stokes_component_index] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
            assert np.allclose(
                eta_s_all[stokes_component_index] / scale,
                eta_s_all_angles_magnetic[stokes_component_index] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
            assert np.allclose(
                eta_a_all[stokes_component_index] / scale,
                eta_a_all_angles_magnetic_rho[stokes_component_index] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
            assert np.allclose(
                eta_s_all[stokes_component_index] / scale,
                eta_s_all_angles_magnetic_rho[stokes_component_index] / scale,
                atol=1e-10,
                rtol=1e-10,
            )
