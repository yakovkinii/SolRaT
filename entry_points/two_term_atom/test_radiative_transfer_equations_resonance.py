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
        # (10.127)
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
            energy_cmm1=220_001,
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
        radiation_tensor.fill_isotropic(I0)
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
        )
        eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
        eta_s_analytic = radiative_transfer_coefficients.eta_s_analytic_resonance(rho=rho, stokes_component_index=0)
        assert (abs(eta_sI - eta_s_analytic) < 1e-10).all()

        eta_sQ = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=1)
        eta_sQ_analytic = radiative_transfer_coefficients.eta_s_analytic_resonance(rho=rho, stokes_component_index=1)
        assert (abs(eta_sQ - eta_sQ_analytic) < 1e-10).all()

        eta_sU = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=2)
        eta_sU_analytic = radiative_transfer_coefficients.eta_s_analytic_resonance(rho=rho, stokes_component_index=2)
        assert (abs(eta_sU - eta_sU_analytic) < 1e-10).all()

        eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)
        eta_sV_analytic = radiative_transfer_coefficients.eta_s_analytic_resonance(rho=rho, stokes_component_index=3)
        assert (abs(eta_sV - eta_sV_analytic) < 1e-10).all()
