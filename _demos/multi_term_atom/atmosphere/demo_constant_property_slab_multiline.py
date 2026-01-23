"""
Multiple spectral lines within constant property slab formalism
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from src.gui.plots.plot_stokes_profiles import StokesPlotter_IV, StokesPlotter_IV_IpmV
from src.multi_term_atom.atmosphere.constant_property_slab import ConstantPropertySlab
from src.multi_term_atom.atmosphere.multi_slab_atmosphere import MultiSlabAtmosphere
from src.multi_term_atom.atmosphere.utils import (
    plot_stokes_IQUV,
    radiation_tensor_NLTE_n_w_parametrized,
)
from src.multi_term_atom.atomic_data.NiI import create_5434_MnFeNi_context


def demo_constant_property_slab_multiline():
    """
    Demonstrate basic usage of ConstantPropertySlab for He I D3 line synthesis.
    """
    logging_config.init(logging.INFO)

    context_Mn, context_Fe, context_Ni = create_5434_MnFeNi_context(lambda_range_A=1.5, lambda_resolution_A=1e-3)

    radiation_tensor_Mn = radiation_tensor_NLTE_n_w_parametrized(context_Mn, h_arcsec=10)
    radiation_tensor_Fe = radiation_tensor_NLTE_n_w_parametrized(context_Fe, h_arcsec=10)
    radiation_tensor_Ni = radiation_tensor_NLTE_n_w_parametrized(context_Ni, h_arcsec=10)

    # Test different magnetic field strengths
    plotter = StokesPlotter_IV_IpmV()

    tau1 = 1.0  # Optical depth
    tau2 = 0.0  # Optical depth
    params1 = {
        "chi": 0,  # Line-of-sight angle
        "theta": 0,  # Inclination angle
        "gamma": 0,  # Azimuth angle
        "magnetic_field_gauss": 2000,  # Magnetic field strength
        "chi_B": 0,  # Magnetic field azimuth
        "theta_B": 0,  # Magnetic field inclination
        "delta_v_thermal_cm_sm1": 500_000,  # Thermal velocity
        "initial_stokes": 1.0,  # Continuum intensity
        "temperature_K": 6000,
        "continuum_opacity": 0,
    }
    params2 = {
        "chi": 0,  # Line-of-sight angle
        "theta": 0,  # Inclination angle
        "gamma": 0,  # Azimuth angle
        "magnetic_field_gauss": -20000,  # Magnetic field strength
        "chi_B": 0,  # Magnetic field azimuth
        "theta_B": 0,  # Magnetic field inclination
        "delta_v_thermal_cm_sm1": 250_000,  # Thermal velocity
        "initial_stokes": 1.0,  # Continuum intensity
        "temperature_K": 5000,
        "continuum_opacity": 0.1,
    }

    # Create constant property slab
    Mn_tau_coef = 0.5  # 3e-3
    slab_Mn = MultiSlabAtmosphere(
        ConstantPropertySlab(
            tau=tau1 * Mn_tau_coef,
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            j_constrained=True,
            **params1,
        ),
        ConstantPropertySlab(
            tau=tau2 * Mn_tau_coef,
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            j_constrained=True,
            **params2,
        ),
    )

    slab_Fe = MultiSlabAtmosphere(
        ConstantPropertySlab(
            tau=tau1,
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            j_constrained=True,
            **params1,
        ),
        ConstantPropertySlab(
            tau=tau2,
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            j_constrained=True,
            **params2,
        ),
    )
    Ni_tau_coef = 0.5  # 8e-4
    slab_Ni = MultiSlabAtmosphere(
        ConstantPropertySlab(
            tau=tau1 * Ni_tau_coef,
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            j_constrained=True,
            **params1,
        ),
        ConstantPropertySlab(
            tau=tau2 * Ni_tau_coef,
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            j_constrained=True,
            **params2,
        ),
    )

    # Solve radiative transfer
    stokes_Mn = slab_Mn.forward_sequential()
    stokes_Fe = slab_Fe.forward_sequential()
    stokes_Ni = slab_Ni.forward_sequential()

    # plotter.add(
    #     lambda_A=context_Mn.lambda_A,
    #     reference_lambda_A=1.5, #context_Mn.reference_lambda_A,
    #     stokes_I=stokes_Mn.I + stokes_Fe.I + stokes_Ni.I - 2,
    #     stokes_V=stokes_Mn.V + stokes_Fe.V + stokes_Ni.V,
    # )
    plotter.add(
        lambda_A=context_Mn.lambda_A,
        reference_lambda_A=1.5,
        stokes_I=stokes_Mn.I,
        stokes_V=stokes_Mn.V,
        label="Mn I 5432",
    )
    plotter.add(
        lambda_A=context_Fe.lambda_A,
        reference_lambda_A=1.5,
        stokes_I=stokes_Fe.I,
        stokes_V=stokes_Fe.V,
        label="Fe I 5434",
    )
    plotter.add(
        lambda_A=context_Ni.lambda_A,
        reference_lambda_A=1.5,
        stokes_I=stokes_Ni.I,
        stokes_V=stokes_Ni.V,
        label="Ni I 5435",
    )
    # Plot results
    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_multiline()
