"""
Demo for ConstantPropertySlab: Basic radiative transfer through a homogeneous atmosphere.

This demo shows how to use the ConstantPropertySlab class to compute Stokes profiles
for the He I D3 line through a slab with constant atmospheric properties.
"""

import logging
import numpy as np
from yatools import logging_config

from src.multi_term_atom.object.multi_term_atom_context import create_he_i_d3_context
from src.multi_term_atom.atmosphere.constant_property_slab import ConstantPropertySlab
from src.multi_term_atom.atmosphere.utils import radiation_tensor_NLTE_n_w_parametrized, plot_stokes_IQUV
import matplotlib.pyplot as plt

def demo_constant_property_slab():
    """
    Demonstrate basic usage of ConstantPropertySlab for He I D3 line synthesis.
    """
    logging_config.init(logging.INFO)

    print("=== ConstantPropertySlab Demo ===")
    print("Computing He I D3 Stokes profiles through a constant property slab...")

    # Create atomic context for He I D3
    context = create_he_i_d3_context(lambda_range_A=0.8, lambda_resolution_A=5e-4)
    print(f"Created He I D3 context with {len(context.lambda_A)} wavelength points")

    # Create radiation tensor (solar limb conditions)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)
    print("Created NLTE radiation tensor for h=10 arcsec above limb")

    # Test different magnetic field strengths
    magnetic_fields = [0, 1000, 3000, 5000]  # Gauss
    stokes_results = []

    for B_field in magnetic_fields:
        print(f"Computing for B = {B_field} G...")

        # Create constant property slab
        slab = ConstantPropertySlab(
            multi_term_atom_context=context,
            radiation_tensor=radiation_tensor,
            tau=1.5,                           # Optical depth
            chi=0,                            # Line-of-sight angle
            theta=0,                          # Inclination angle
            gamma=0,                          # Azimuth angle
            magnetic_field_gauss=B_field,     # Magnetic field strength
            chi_B=0,                          # Magnetic field azimuth
            theta_B=0,                        # Magnetic field inclination
            delta_v_thermal_cm_sm1=40000,     # Thermal velocity
            initial_stokes=1.0                # Continuum intensity
        )

        # Solve radiative transfer
        stokes = slab.forward()
        stokes_results.append(stokes)

    # Plot results
    print("Plotting results...")
    labels = [f"B = {B} G" for B in magnetic_fields]
    colors = ['blue', 'green', 'orange', 'red']


    fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=(10, 8))
    fig.suptitle("He I D3 Stokes Profiles: Constant Property Slab", fontsize=14)

    for stokes, label, color in zip(stokes_results, labels, colors):
        plot_stokes_IQUV(stokes, label, context.reference_lambda_A,
                        show=False, axs=axs, normalize=True, color=color)

    plt.show()

    # Print some diagnostics
    print("\n=== Results Summary ===")
    for i, (B, stokes) in enumerate(zip(magnetic_fields, stokes_results)):
        line_center_idx = len(stokes.I) // 2
        I_center = stokes.I[line_center_idx]
        V_peak = np.max(np.abs(stokes.V))
        print(f"B = {B:4d} G: I_center = {I_center:.4f}, V_peak = {V_peak:.4f}")



if __name__ == "__main__":
    demo_constant_property_slab()
