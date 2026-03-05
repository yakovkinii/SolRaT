import logging
import unittest

import numpy as np
from yatools import logging_config

from solrat.engine.functions.special import pseudo_hash
from solrat.multi_term_atom.atmosphere.constant_property_slab import (
    ConstantPropertySlabAtmosphere,
)
from solrat.multi_term_atom.atmosphere.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.multi_term_atom.atomic_data.MnI import get_Mn_I_5432_data
from solrat.multi_term_atom.object.angles import Angles
from solrat.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from solrat.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from solrat.multi_term_atom.object.radiation_tensor import RadiationTensor
from solrat.multi_term_atom.object.stokes import Stokes
from solrat.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEELTE


class TestRadiativeTransferEquations(unittest.TestCase):
    def test_radiative_transfer_equations(self):
        """
        Demonstrate basic usage of ConstantPropertySlab for He I D3 line synthesis.
        """
        logging_config.init(logging.INFO)

        level_registry_Mn, transition_registry_Mn, reference_lambda_A_Mn, _, atomic_mass_amu_Mn = get_Mn_I_5432_data()

        lambda_A_Mn = np.arange(reference_lambda_A_Mn + 1.5 - 0.5, reference_lambda_A_Mn + 1.1 + 1, 1e-3)

        context_Mn = MultiTermAtomContext(
            level_registry=level_registry_Mn,
            transition_registry=transition_registry_Mn,
            statistical_equilibrium_equations=MultiTermAtomSEELTE(
                level_registry=level_registry_Mn,
            ),
            lambda_A=lambda_A_Mn,
            reference_lambda_A=reference_lambda_A_Mn,
            atomic_mass_amu=atomic_mass_amu_Mn,
            j_constrained=True,
        )

        radiation_tensor_Mn = RadiationTensor(context_Mn.transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=0)

        # Test different magnetic field strengths

        angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

        # Atmosphere parameters:
        atmosphere1 = {
            "magnetic_field_gauss": 1000,
            "temperature_K": 5000,
            "delta_v_turbulent_cm_sm1": 1000_00,
            "macroscopic_velocity_cm_sm1": 0,
            "voigt_a": 0,
        }

        initial_stokes_Mn = Stokes.from_BP(nu_sm1=context_Mn.nu, temperature_K=5700)

        slab1_continuum_delta_tau = 0.01

        atmosphere_Mn = MultiSlabAtmosphere(
            ConstantPropertySlabAtmosphere(
                multi_term_atom_context=context_Mn,
                radiation_tensor=radiation_tensor_Mn,
                line_delta_tau=0.3,
                continuum_delta_tau=slab1_continuum_delta_tau,
                angles=angles,
                atmosphere_parameters=AtmosphereParameters(atomic_mass_amu=atomic_mass_amu_Mn, **atmosphere1),
            ),
        )

        stokes_Mn = atmosphere_Mn.forward(initial_stokes=initial_stokes_Mn)

        # Check that the result did not change from previous runs
        last_run_hash = 8.085295570214492e-05
        new_hash = pseudo_hash(stokes_Mn.I, stokes_Mn.Q, stokes_Mn.U, stokes_Mn.V)
        logging.info(new_hash)
        logging.info(last_run_hash)
        assert np.abs((last_run_hash - new_hash) / last_run_hash) < 1e-8
