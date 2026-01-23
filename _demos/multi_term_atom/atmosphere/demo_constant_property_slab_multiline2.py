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
from src.multi_term_atom.object.multi_term_atom_context import (
    create_5434_MnFeNi_context,
    create_he_i_d3_context,
)


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

    common_params = {
        "chi": 0.0,  # LOS azimuth
        "theta": 0.0,  # LOS inclination (mu = 1)
        "gamma": 0.0,  # LOS rotation angle for Stokes reference
        "initial_stokes": 1.0,  # Continuum intensity
    }

    slab1_params = {
        "temperature_K": 4500,
        "continuum_opacity": 1,  # Δτ_c,1
        "magnetic_field_gauss": 2500,
        "theta_B": 0.0,  # vertical field
        "chi_B": 0.0,
        "delta_v_thermal_cm_sm1": 2.5e5,  # you can override per slab if needed
    }
    tau_line_slab1 = {
        "Fe5434": 4.0,
        "Mn5432": 3.0,
        "Ni5435": 2.5,
    }

    slab2_params = {
        "temperature_K": 4100,
        "continuum_opacity": 0.4,  # Δτ_c,2
        "magnetic_field_gauss": 3000,
        "theta_B": 0.0,
        "chi_B": 0.0,
        "delta_v_thermal_cm_sm1": 2.5e5,
    }
    tau_line_slab2 = {
        "Fe5434": 3.0,
        "Mn5432": 1.0,
        "Ni5435": 0.8,
    }

    slab3_params = {
        "temperature_K": 3800,
        "continuum_opacity": 0.2,  # Δτ_c,3 (thin)
        "magnetic_field_gauss": 20000,  # HIGH FIELD
        "theta_B": 0.0,
        "chi_B": 0.0,
        "delta_v_thermal_cm_sm1": 2.5e5,
    }
    tau_line_slab3 = {
        "Fe5434": 1.0,  # Fe sees this slab strongly
        "Mn5432": 0.10,  # Mn nearly blind
        "Ni5435": 0.05,  # Ni nearly blind
    }

    slab_Mn = MultiSlabAtmosphere(
        ConstantPropertySlab(
            tau=tau_line_slab1["Mn5432"],
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            j_constrained=True,
            **common_params,
            **slab1_params,
        ),
        ConstantPropertySlab(
            tau=tau_line_slab2["Mn5432"],
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            j_constrained=True,
            **common_params,
            **slab2_params,
        ),
        ConstantPropertySlab(
            tau=tau_line_slab3["Mn5432"],
            multi_term_atom_context=context_Mn,
            radiation_tensor=radiation_tensor_Mn,
            j_constrained=True,
            **common_params,
            **slab3_params,
        ),
    )

    slab_Fe = MultiSlabAtmosphere(
        ConstantPropertySlab(
            tau=tau_line_slab1["Fe5434"],
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            j_constrained=True,
            **common_params,
            **slab1_params,
        ),
        ConstantPropertySlab(
            tau=tau_line_slab2["Fe5434"],
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            j_constrained=True,
            **common_params,
            **slab2_params,
        ),
        ConstantPropertySlab(
            tau=tau_line_slab3["Fe5434"],
            multi_term_atom_context=context_Fe,
            radiation_tensor=radiation_tensor_Fe,
            j_constrained=True,
            **common_params,
            **slab3_params,
        ),
    )
    Ni_tau_coef = 0.5  # 8e-4
    slab_Ni = MultiSlabAtmosphere(
        ConstantPropertySlab(
            tau=tau_line_slab1["Ni5435"],
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            j_constrained=True,
            **common_params,
            **slab1_params,
        ),
        ConstantPropertySlab(
            tau=tau_line_slab2["Ni5435"],
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            j_constrained=True,
            **common_params,
            **slab2_params,
        ),
        ConstantPropertySlab(
            tau=tau_line_slab3["Ni5435"],
            multi_term_atom_context=context_Ni,
            radiation_tensor=radiation_tensor_Ni,
            j_constrained=True,
            **common_params,
            **slab3_params,
        ),
    )

    # Solve radiative transfer
    # stokes_Mn = slab_Mn.forward_sequential()
    stokes_Fe, stokes_list_Fe = slab_Fe.forward_sequential()
    # stokes_Ni = slab_Ni.forward_sequential()

    # plotter.add(
    #     lambda_A=context_Mn.lambda_A,
    #     reference_lambda_A=1.5, #context_Mn.reference_lambda_A,
    #     stokes_I=stokes_Mn.I + stokes_Fe.I + stokes_Ni.I - 2,
    #     stokes_V=stokes_Mn.V + stokes_Fe.V + stokes_Ni.V,
    # )
    # plotter.add(
    #     lambda_A=context_Mn.lambda_A,
    #     reference_lambda_A=1.5,
    #     stokes_I=stokes_Mn.I,
    #     stokes_V=stokes_Mn.V,
    #     label="Mn I 5432",
    # )

    for i, st in enumerate(stokes_list_Fe):
        plotter.add(
            lambda_A=context_Fe.lambda_A,
            reference_lambda_A=1.5,
            stokes_I=st.I,
            stokes_V=st.V,
            label=f"Fe I 5434 (after layer{i+1})",
        )

    plotter.add(
        lambda_A=context_Fe.lambda_A,
        reference_lambda_A=1.5,
        stokes_I=stokes_Fe.I,
        stokes_V=stokes_Fe.V,
        label="Fe I 5434",
    )
    # plotter.add(
    #     lambda_A=context_Ni.lambda_A,
    #     reference_lambda_A=1.5,
    #     stokes_I=stokes_Ni.I,
    #     stokes_V=stokes_Ni.V,
    #     label="Ni I 5435",
    # )
    # Plot results
    plotter.show()


if __name__ == "__main__":
    demo_constant_property_slab_multiline()
