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
from src.multi_term_atom.atomic_data.FeI import get_Fe_I_5434_data
from src.multi_term_atom.atomic_data.MnI import get_Mn_I_5432_data
from src.multi_term_atom.atomic_data.NiI import get_Ni_I_5435_data
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
    level_registry_Fe, transition_registry_Fe, reference_lambda_A_Fe, _, atomic_mass_amu_Fe = get_Fe_I_5434_data()
    level_registry_Ni, transition_registry_Ni, reference_lambda_A_Ni, _, atomic_mass_amu_Ni = get_Ni_I_5435_data()

    lambda_A_Mn = np.arange(reference_lambda_A_Mn + 1.5 - 0.5, reference_lambda_A_Mn + 1.5 + 0.5, 1e-3)
    lambda_A_Fe = np.arange(reference_lambda_A_Fe + 1.5 - 0.5, reference_lambda_A_Fe + 1.5 + 0.5, 1e-3)
    lambda_A_Ni = np.arange(reference_lambda_A_Ni + 1.5 - 0.5, reference_lambda_A_Ni + 1.5 + 0.5, 1e-3)

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

    context_Fe = MultiTermAtomContext(
        level_registry=level_registry_Fe,
        transition_registry=transition_registry_Fe,
        statistical_equilibrium_equations=MultiTermAtomSEELTE(
            level_registry=level_registry_Fe,
        ),
        lambda_A=lambda_A_Fe,
        reference_lambda_A=reference_lambda_A_Fe,
        atomic_mass_amu=atomic_mass_amu_Fe,
        j_constrained=True,
    )

    context_Ni = MultiTermAtomContext(
        level_registry=level_registry_Ni,
        transition_registry=transition_registry_Ni,
        statistical_equilibrium_equations=MultiTermAtomSEELTE(
            level_registry=level_registry_Ni,
        ),
        lambda_A=lambda_A_Ni,
        reference_lambda_A=reference_lambda_A_Ni,
        atomic_mass_amu=atomic_mass_amu_Ni,
        j_constrained=True,
    )

    radiation_tensor_Mn = RadiationTensor(context_Mn.transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=0)
    radiation_tensor_Fe = RadiationTensor(context_Fe.transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=0)
    radiation_tensor_Ni = RadiationTensor(context_Ni.transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=0)

    # Test different magnetic field strengths
    plotter = StokesPlotter_IV_IpmV()

    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    # Atmosphere parameters:
    atmosphere1 = {
        "magnetic_field_gauss": 3000,
        "temperature_K": 4500,
        "macroscopic_velocity_cm_sm1": 0,
    }
    atmosphere2 = {
        "magnetic_field_gauss": -15000,
        "temperature_K": 9000,
        "macroscopic_velocity_cm_sm1": 0,
    }

    initial_stokes_Mn = Stokes.from_BP(nu_sm1=context_Mn.nu, temperature_K=5700)
    initial_stokes_Fe = Stokes.from_BP(nu_sm1=context_Fe.nu, temperature_K=5700)
    initial_stokes_Ni = Stokes.from_BP(nu_sm1=context_Ni.nu, temperature_K=5700)

    slab1_continuum_delta_tau = 0.05
    slab2_continuum_delta_tau = 0.005

    atmosphere_Mn = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            line_delta_tau=1.6,
            continuum_delta_tau=slab1_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                atomic_mass_amu=atomic_mass_amu_Mn,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere1,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            line_delta_tau=0.01,
            continuum_delta_tau=slab2_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                atomic_mass_amu=atomic_mass_amu_Mn,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere2,
            ),
        ),
    )

    stokes_Mn = atmosphere_Mn.forward(initial_stokes=initial_stokes_Mn)

    atmosphere_Fe = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            line_delta_tau=2.6,
            continuum_delta_tau=slab1_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                atomic_mass_amu=atomic_mass_amu_Fe,
                delta_v_turbulent_cm_sm1=2000_00,
                voigt_a=0,
                **atmosphere1,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            line_delta_tau=0.02,
            continuum_delta_tau=slab2_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                atomic_mass_amu=atomic_mass_amu_Fe,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere2,
            ),
        ),
    )

    stokes_Fe = atmosphere_Fe.forward(initial_stokes=initial_stokes_Fe)

    atmosphere_Ni = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            line_delta_tau=1.5,
            continuum_delta_tau=slab1_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                atomic_mass_amu=atomic_mass_amu_Ni,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere1,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            line_delta_tau=0.01,
            continuum_delta_tau=slab2_continuum_delta_tau,
            angles=angles,
            atmosphere_parameters=AtmosphereParameters(
                atomic_mass_amu=atomic_mass_amu_Ni,
                delta_v_turbulent_cm_sm1=5000_00,
                voigt_a=0,
                **atmosphere2,
            ),
        ),
    )

    stokes_Ni = atmosphere_Ni.forward(initial_stokes=initial_stokes_Ni)

    plotter.add_stokes(
        lambda_A=np.concat([context_Mn.lambda_A, context_Fe.lambda_A, context_Ni.lambda_A]),
        reference_lambda_A=1.5,
        stokes=Stokes(
            nu=np.concat([stokes_Mn.nu, stokes_Fe.nu, stokes_Ni.nu]),
            I=np.concat([stokes_Mn.I, stokes_Fe.I, stokes_Ni.I]),
            Q=np.concat([stokes_Mn.Q, stokes_Fe.Q, stokes_Ni.Q]),
            U=np.concat([stokes_Mn.U, stokes_Fe.U, stokes_Ni.U]),
            V=np.concat([stokes_Mn.V, stokes_Fe.V, stokes_Ni.V]),
        ),
        stokes_reference=Stokes(
            nu=np.concat([initial_stokes_Mn.nu, initial_stokes_Fe.nu, initial_stokes_Ni.nu]),
            I=np.concat([initial_stokes_Mn.I, initial_stokes_Fe.I, initial_stokes_Ni.I]),
            Q=np.concat([initial_stokes_Mn.Q, initial_stokes_Fe.Q, initial_stokes_Ni.Q]),
            U=np.concat([initial_stokes_Mn.U, initial_stokes_Fe.U, initial_stokes_Ni.U]),
            V=np.concat([initial_stokes_Mn.V, initial_stokes_Fe.V, initial_stokes_Ni.V]),
        ),
        label="RTE with LTE SEE",
    )

    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_multiline()
