import logging
import unittest
from pathlib import Path

import numpy as np
from yatools import logging_config

from src.engine.functions.special import pseudo_hash
from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.atomic_data.HeI import fill_precomputed_He_I_D3_data, get_He_I_D3_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        logging_config.init(logging.INFO)

        # Load the atomic data for He I D3
        level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()

        # The calculation itself needs frequency, but we will display the results in wavelength
        lambda_A = np.arange(reference_lambda_A - 2, reference_lambda_A + 2, 5e-4)
        nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

        angles = Angles(
            chi=np.pi / 5,
            theta=np.pi / 7,
            gamma=np.pi / 9,
            chi_B=np.pi / 3,
            theta_B=np.pi / 5,
        )

        # Set up the statistical equilibrium equations.
        # Do not pre-compute coefficients, load them from file instead for the purpose of this demo.
        see = MultiTermAtomSEE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            precompute=False,
        )
        fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.as_posix())

        # Set up the radiative transfer equations
        # Angles input is optional. But since we know the angles in advance,
        # we provide them here to speed up the calculation.
        rte = MultiTermAtomRTE(
            level_registry=level_registry,
            transition_registry=transition_registry,
            nu=nu,
            angles=angles,
        )

        # Fill the radiation tensor with anisotropic radiation field 10 arcsec from the Sun's apparent surface
        radiation_tensor = (
            RadiationTensor(transition_registry=transition_registry)
            .fill_NLTE_n_w_parametrized(h_arcsec=10)
            .rotate_to_magnetic_frame(
                chi_B=angles.chi_B,
                theta_B=angles.theta_B,
            )
        )

        # Set up the atmosphere parameters
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=1000,
            delta_v_thermal_cm_sm1=1_000_00,
        )

        # Construct all equations for rho
        see.add_all_equations(
            atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
        )

        # Solve all equations for rho
        rho = see.get_solution_direct()

        # get RT coefficients. They are complex: eta = real(eta_rho), rho = imag(eta_rho)
        eta_rho_sI, eta_rho_sQ, eta_rho_sU, eta_rho_sV = rte.eta_rho_s(
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
        )

        # Check that the result did not change from previous runs
        assert pseudo_hash(eta_rho_sI, eta_rho_sQ, eta_rho_sU, eta_rho_sV) == 2.313240321824078e-16
