import logging
import unittest

import numpy as np
from yatools import logging_config

from src.core.physics.functions import get_planck_BP
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        logging_config.init(logging.INFO)

        term_registry = TermRegistry()
        term_registry.register_term(
            beta="1s",
            L=0,
            S=0.5,
            J=0.5,
            energy_cmm1=200_000,
        )
        term_registry.register_term(
            beta="2p",
            L=1,
            S=0.5,
            J=0.5,
            energy_cmm1=220_000,
        )
        term_registry.register_term(
            beta="2p",
            L=1,
            S=0.5,
            J=1.5,
            energy_cmm1=220_000,
        )
        term_registry.validate()

        nu = np.arange(5.995e14, 5.997e14, 1e9)  # Hz

        transition_registry = TransitionRegistry()
        transition_registry.register_transition_from_a_ul(
            level_upper=term_registry.get_level(beta="2p", L=1, S=0.5),
            level_lower=term_registry.get_level(beta="1s", L=0, S=0.5),
            einstein_a_ul_sm1=0.7e8,
        )

        atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=5_000_00)
        radiation_tensor = RadiationTensor(transition_registry=transition_registry)
        I0 = get_planck_BP(nu_sm1=nu, T_K=5000)
        radiation_tensor.fill_NLTE_near_isotropic(I0)
        atom = TwoTermAtom(
            term_registry=term_registry,
            transition_registry=transition_registry,
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor=radiation_tensor,
            disable_r_s=True,
            disable_n=True,
            n_frequencies=len(nu),
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
        eta_s_analytic = radiative_transfer_coefficients.eta_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=0
        )
        assert (abs(eta_sI - eta_s_analytic) < 1e-10).all()

        eta_sQ = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=1)
        eta_sQ_analytic = radiative_transfer_coefficients.eta_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=1
        )
        assert (abs(eta_sQ - eta_sQ_analytic) < 1e-10).all()

        eta_sU = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=2)
        eta_sU_analytic = radiative_transfer_coefficients.eta_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=2
        )
        assert (abs(eta_sU - eta_sU_analytic) < 1e-10).all()

        eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)
        eta_sV_analytic = radiative_transfer_coefficients.eta_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=3
        )
        assert (abs(eta_sV - eta_sV_analytic) < 1e-10).all()

        eta_aI = radiative_transfer_coefficients.eta_a(rho=rho, stokes_component_index=0)
        eta_a_analytic = radiative_transfer_coefficients.eta_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=0
        )
        assert (abs(eta_aI - eta_a_analytic) < 1e-10).all()

        eta_aQ = radiative_transfer_coefficients.eta_a(rho=rho, stokes_component_index=1)
        eta_aQ_analytic = radiative_transfer_coefficients.eta_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=1
        )
        assert (abs(eta_aQ - eta_aQ_analytic) < 1e-10).all()

        eta_aU = radiative_transfer_coefficients.eta_a(rho=rho, stokes_component_index=2)
        eta_aU_analytic = radiative_transfer_coefficients.eta_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=2
        )
        assert (abs(eta_aU - eta_aU_analytic) < 1e-10).all()

        eta_aV = radiative_transfer_coefficients.eta_a(rho=rho, stokes_component_index=3)
        eta_aV_analytic = radiative_transfer_coefficients.eta_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=3
        )
        assert (abs(eta_aV - eta_aV_analytic) < 1e-10).all()

        rho_sI = radiative_transfer_coefficients.rho_s(rho=rho, stokes_component_index=0)
        rho_s_analytic = radiative_transfer_coefficients.rho_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=0
        )
        assert (abs(rho_sI - rho_s_analytic) < 1e-10).all()

        rho_sQ = radiative_transfer_coefficients.rho_s(rho=rho, stokes_component_index=1)
        rho_sQ_analytic = radiative_transfer_coefficients.rho_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=1
        )
        assert (abs(rho_sQ - rho_sQ_analytic) < 1e-10).all()

        rho_sU = radiative_transfer_coefficients.rho_s(rho=rho, stokes_component_index=2)
        rho_sU_analytic = radiative_transfer_coefficients.rho_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=2
        )
        assert (abs(rho_sU - rho_sU_analytic) < 1e-10).all()

        rho_sV = radiative_transfer_coefficients.rho_s(rho=rho, stokes_component_index=3)
        rho_sV_analytic = radiative_transfer_coefficients.rho_s_no_field_no_fine_structure(
            rho=rho, stokes_component_index=3
        )
        assert (abs(rho_sV - rho_sV_analytic) < 1e-10).all()

        rho_aI = radiative_transfer_coefficients.rho_a(rho=rho, stokes_component_index=0)
        rho_a_analytic = radiative_transfer_coefficients.rho_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=0
        )
        assert (abs(rho_aI - rho_a_analytic) < 1e-10).all()

        rho_aQ = radiative_transfer_coefficients.rho_a(rho=rho, stokes_component_index=1)
        rho_aQ_analytic = radiative_transfer_coefficients.rho_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=1
        )
        assert (abs(rho_aQ - rho_aQ_analytic) < 1e-10).all()

        rho_aU = radiative_transfer_coefficients.rho_a(rho=rho, stokes_component_index=2)
        rho_aU_analytic = radiative_transfer_coefficients.rho_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=2
        )
        assert (abs(rho_aU - rho_aU_analytic) < 1e-10).all()

        rho_aV = radiative_transfer_coefficients.rho_a(rho=rho, stokes_component_index=3)
        rho_aV_analytic = radiative_transfer_coefficients.rho_a_no_field_no_fine_structure(
            rho=rho, stokes_component_index=3
        )
        assert (abs(rho_aV - rho_aV_analytic) < 1e-10).all()

        epsilon_I = radiative_transfer_coefficients.epsilon(eta_s=eta_sI, nu=nu)
        epsilon_I_analytic = radiative_transfer_coefficients.epsilon(eta_s=eta_s_analytic, nu=nu)
        assert (abs(epsilon_I - epsilon_I_analytic) < 1e-10).all()

        epsilon_Q = radiative_transfer_coefficients.epsilon(eta_s=eta_sQ, nu=nu)
        epsilon_Q_analytic = radiative_transfer_coefficients.epsilon(eta_s=eta_sQ_analytic, nu=nu)
        assert (abs(epsilon_Q - epsilon_Q_analytic) < 1e-10).all()

        epsilon_U = radiative_transfer_coefficients.epsilon(eta_s=eta_sU, nu=nu)
        epsilon_U_analytic = radiative_transfer_coefficients.epsilon(eta_s=eta_sU_analytic, nu=nu)
        assert (abs(epsilon_U - epsilon_U_analytic) < 1e-10).all()

        epsilon_V = radiative_transfer_coefficients.epsilon(eta_s=eta_sV, nu=nu)
        epsilon_V_analytic = radiative_transfer_coefficients.epsilon(eta_s=eta_sV_analytic, nu=nu)
        assert (abs(epsilon_V - epsilon_V_analytic) < 1e-10).all()
