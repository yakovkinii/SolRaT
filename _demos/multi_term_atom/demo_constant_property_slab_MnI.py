"""
Multiple spectral lines within constant property slab formalism
"""

import logging

import numpy as np
from yatools import logging_config

from src.gui.plots.plot_stokes_profiles import StokesPlotter_IV_IpmV
from src.multi_term_atom.atmosphere.constant_property_slab import (
    ConstantPropertySlabAtmosphere,
)
from src.multi_term_atom.atmosphere.multi_slab_atmosphere import MultiSlabAtmosphere
from src.multi_term_atom.atomic_data.MnI import get_Mn_I_5432_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEELTE


def demo_constant_property_slab_multiline():
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
    plotter = StokesPlotter_IV_IpmV()

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

    plotter.add_stokes(
        lambda_A=np.concat([context_Mn.lambda_A]),
        reference_lambda_A=1.5,
        stokes=Stokes(
            nu=np.concat([stokes_Mn.nu]),
            I=np.concat([stokes_Mn.I]),
            Q=np.concat([stokes_Mn.Q]),
            U=np.concat([stokes_Mn.U]),
            V=np.concat([stokes_Mn.V]),
        ),
        stokes_reference=Stokes(
            nu=np.concat([initial_stokes_Mn.nu]),
            I=np.concat([initial_stokes_Mn.I]),
            Q=np.concat([initial_stokes_Mn.Q]),
            U=np.concat([initial_stokes_Mn.U]),
            V=np.concat([initial_stokes_Mn.V]),
        ),
        label="RTE with LTE SEE",
    )

    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_multiline()
