import logging
import unittest

import numpy as np
from yatools import logging_config

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.constants import atomic_mass_unit_g, kB_erg_Km1
from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
from solrat.engine.functions.special import pseudo_hash


class TestRadiativeTransferEquationsD3(unittest.TestCase):
    def test_radiative_transfer_equations_d3(self):
        logging_config.init(logging.INFO)

        model = PreconfiguredModels.multi_term_atom_HeID3()
        reference_lambda = model.config.reference_lambda_A

        # The calculation itself needs frequency, but we will display the results in wavelength
        lambda_A = np.arange(reference_lambda - 2, reference_lambda + 2, 5e-4)
        nu = lambda_A_to_frequency_hz(lambda_A)

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
        rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)

        # Fill the radiation tensor with anisotropic radiation field 10 arcsec from the Sun's apparent surface
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(
            h_arcsec=10,
        )

        # Set up the atmosphere parameters
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=1000,
            temperature_K=1_000_00**2 / kB_erg_Km1 / 2 * 4 * atomic_mass_unit_g,
        )
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor.rotate_to_magnetic_frame(angles=angles),
        )

        rho = see.get_solution()

        # get RT coefficients. They are complex: eta = real(eta_rho), rho = imag(eta_rho)
        eta_rho_sI = rte.calculate_eta_rho_s(
            stokes_component_index=0, rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_rho_sQ = rte.calculate_eta_rho_s(
            stokes_component_index=1, rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_rho_sU = rte.calculate_eta_rho_s(
            stokes_component_index=2, rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles
        )
        eta_rho_sV = rte.calculate_eta_rho_s(
            stokes_component_index=3, rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles
        )

        # Check that the result did not change from previous runs
        last_run_hash = 2.3137071959665785e-16
        new_hash = pseudo_hash(eta_rho_sI, eta_rho_sQ, eta_rho_sU, eta_rho_sV)
        logging.info(new_hash)
        logging.info(last_run_hash)
        assert np.abs((last_run_hash - new_hash) / last_run_hash) < 1e-8
