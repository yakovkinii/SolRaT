import logging
import unittest

import numpy as np
from yatools import logging_config

from src.two_term_atom.atomic_data.mock import get_mock_atom_data
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        # (10.127)
        logging_config.init(logging.INFO)
        term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()
        nu = np.arange(reference_nu_sm1 - 1e11, reference_nu_sm1 + 1e11, 1e9)  # Hz

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=5_000_00)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_w(h_arcsec=30)
        atom = TwoTermAtom(
            term_registry=term_registry,
            transition_registry=transition_registry,
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor=radiation_tensor,
            disable_r_s=True,
            disable_n=True,
        )

        atom.add_all_equations()
        rho = atom.get_solution_direct()

        radiative_transfer_coefficients = RadiativeTransferCoefficients(
            atmosphere_parameters=atmosphere_parameters,
            transition_registry=transition_registry,
            nu=nu,
            theta=np.pi / 8,
            gamma=np.pi / 8,
            chi=np.pi / 8,
        )
        eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
        eta_s_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=0)
        assert (abs(eta_sI - eta_s_analytic) < 1e-10).all()

        eta_sQ = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=1)
        eta_sQ_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=1)
        assert (abs(eta_sQ - eta_sQ_analytic) < 1e-10).all()

        eta_sU = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=2)
        eta_sU_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=2)
        assert (abs(eta_sU - eta_sU_analytic) < 1e-10).all()

        eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)
        eta_sV_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=3)
        assert (abs(eta_sV - eta_sV_analytic) < 1e-10).all()
