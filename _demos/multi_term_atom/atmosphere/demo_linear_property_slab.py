"""
Demo for LinearPropertySlab: Radiative transfer through atmospheres with linear gradients.

This demo shows how to use the LinearPropertySlab class to compute Stokes profiles
for the He I D3 line through atmospheres with linearly varying properties.
"""

import logging
import numpy as np
from yatools import logging_config

from src.multi_term_atom.object.multi_term_atom_context import create_he_i_d3_context
from src.multi_term_atom.atmosphere.constant_property_slab import ConstantPropertySlab
from src.multi_term_atom.atmosphere.linear_property_slab import LinearPropertySlab
from src.multi_term_atom.atmosphere.utils import radiation_tensor_NLTE_n_w_parametrized, plot_stokes_comparison


def demo_linear_vs_constant():
    """
    Compare LinearPropertySlab with equivalent ConstantPropertySlab.
    """
    logging_config.init(logging.INFO)

    print("=== LinearPropertySlab vs ConstantPropertySlab Demo ===")
    print("Comparing linear gradient with constant property approximations...")

    # Create atomic context
    context = create_he_i_d3_context(lambda_range_A=0.8, lambda_resolution_A=5e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Define gradient parameters
    B_top = 1000      # Gauss at top
    B_bottom = 4000   # Gauss at bottom
    v_th_top = 30000  # cm/s at top
    v_th_bottom = 50000  # cm/s at bottom
    tau_total = 2.0

    print(f"Gradient: B = {B_top} → {B_bottom} G, v_th = {v_th_top/1000:.0f} → {v_th_bottom/1000:.0f} km/s")

    # 1. Linear property slab
    linear_slab = LinearPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau_total=tau_total,
        magnetic_field_gauss_top=B_top,
        magnetic_field_gauss_bottom=B_bottom,
        delta_v_thermal_cm_sm1_top=v_th_top,
        delta_v_thermal_cm_sm1_bottom=v_th_bottom,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        n_sub_slabs=20,  # High resolution for smooth gradient
        initial_stokes=1.0
    )

    print("Computing linear gradient solution...")
    stokes_linear = linear_slab.forward()

    # 2. Constant property slab with average values
    B_avg = (B_top + B_bottom) / 2
    v_th_avg = (v_th_top + v_th_bottom) / 2

    constant_slab = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=tau_total,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=B_avg,
        delta_v_thermal_cm_sm1=v_th_avg,
        initial_stokes=1.0
    )

    print("Computing constant property solution (average values)...")
    stokes_constant = constant_slab.forward()

    # 3. Constant property slab with top values
    constant_top_slab = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=tau_total,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=B_top,
        delta_v_thermal_cm_sm1=v_th_top,
        initial_stokes=1.0
    )

    print("Computing constant property solution (top values)...")
    stokes_top = constant_top_slab.forward()

    # 4. Constant property slab with bottom values
    constant_bottom_slab = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=tau_total,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=B_bottom,
        delta_v_thermal_cm_sm1=v_th_bottom,
        initial_stokes=1.0
    )

    print("Computing constant property solution (bottom values)...")
    stokes_bottom = constant_bottom_slab.forward()

    # Plot comparison
    stokes_list = [stokes_linear, stokes_constant, stokes_top, stokes_bottom]
    labels = ["Linear gradient", "Constant (avg)", "Constant (top)", "Constant (bottom)"]
    colors = ['red', 'blue', 'green', 'orange']

    plot_stokes_comparison(
        stokes_list, labels, context.reference_lambda_A,
        title="Linear Gradient vs Constant Property Comparison"
    )


def demo_gradient_effects():
    """
    Demonstrate effects of different gradient strengths.
    """
    print("\n=== Gradient Strength Effects Demo ===")
    print("Studying effects of gradient strength on Stokes profiles...")

    # Create context
    context = create_he_i_d3_context(lambda_range_A=0.6, lambda_resolution_A=3e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Base parameters
    B_base = 2000
    v_th_base = 40000
    tau_total = 1.5

    # Different gradient strengths
    gradient_factors = [0.0, 0.5, 1.0, 2.0]  # Multiplier for gradient strength

    stokes_results = []
    labels = []

    for factor in gradient_factors:
        # Calculate top and bottom values
        delta_B = B_base * factor * 0.5  # ±50% variation
        delta_v = v_th_base * factor * 0.25  # ±25% variation

        B_top = B_base - delta_B
        B_bottom = B_base + delta_B
        v_th_top = v_th_base - delta_v
        v_th_bottom = v_th_base + delta_v

        if factor == 0.0:
            # Use constant slab for zero gradient
            slab = ConstantPropertySlab(
                multi_term_atom_context=context,
                radiation_tensor=radiation_tensor,
                tau=tau_total,
                chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
                magnetic_field_gauss=B_base,
                delta_v_thermal_cm_sm1=v_th_base,
                initial_stokes=1.0
            )
            labels.append("No gradient")
        else:
            slab = LinearPropertySlab(
                multi_term_atom_context=context,
                radiation_tensor=radiation_tensor,
                tau_total=tau_total,
                magnetic_field_gauss_top=B_top,
                magnetic_field_gauss_bottom=B_bottom,
                delta_v_thermal_cm_sm1_top=v_th_top,
                delta_v_thermal_cm_sm1_bottom=v_th_bottom,
                chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
                n_sub_slabs=15,
                initial_stokes=1.0
            )
            labels.append(f"Gradient factor {factor}")

        print(f"Computing gradient factor {factor}...")
        stokes = slab.forward()
        stokes_results.append(stokes)

    # Plot results
    plot_stokes_comparison(
        stokes_results, labels, context.reference_lambda_A,
        title="Effect of Gradient Strength on He I D3 Profiles"
    )


def demo_resolution_convergence():
    """
    Demonstrate convergence with number of sub-slabs.
    """
    print("\n=== Resolution Convergence Demo ===")
    print("Testing convergence with number of sub-slabs...")

    # Create context
    context = create_he_i_d3_context(lambda_range_A=0.4, lambda_resolution_A=2e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Strong gradient for testing convergence
    B_top = 500
    B_bottom = 5000  # 10x increase
    v_th_top = 25000
    v_th_bottom = 55000
    tau_total = 1.8

    # Different resolutions
    n_sub_slabs_list = [2, 5, 10, 20, 40]

    stokes_results = []
    labels = []

    for n_subs in n_sub_slabs_list:
        print(f"Computing with {n_subs} sub-slabs...")

        slab = LinearPropertySlab(
            multi_term_atom_context=context,
            radiation_tensor=radiation_tensor,
            tau_total=tau_total,
            magnetic_field_gauss_top=B_top,
            magnetic_field_gauss_bottom=B_bottom,
            delta_v_thermal_cm_sm1_top=v_th_top,
            delta_v_thermal_cm_sm1_bottom=v_th_bottom,
            chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
            n_sub_slabs=n_subs,
            initial_stokes=1.0
        )

        stokes = slab.forward()
        stokes_results.append(stokes)
        labels.append(f"{n_subs} sub-slabs")

    # Plot convergence
    plot_stokes_comparison(
        stokes_results, labels, context.reference_lambda_A,
        title="Convergence Test: Number of Sub-slabs"
    )

    # Print convergence metrics
    print("\n=== Convergence Analysis ===")
    reference_stokes = stokes_results[-1]  # Highest resolution as reference

    for i, (n_subs, stokes) in enumerate(zip(n_sub_slabs_list[:-1], stokes_results[:-1])):
        # Compare line center intensity
        center_idx = len(stokes.I) // 2
        I_diff = abs(stokes.I[center_idx] - reference_stokes.I[center_idx])
        V_rms = np.sqrt(np.mean((stokes.V - reference_stokes.V)**2))

        print(f"{n_subs:2d} sub-slabs: ΔI_center = {I_diff:.6f}, V_RMS_diff = {V_rms:.6f}")


if __name__ == "__main__":
    demo_linear_vs_constant()
    demo_gradient_effects()
    demo_resolution_convergence()

    print("\n=== Demo Complete ===")
    print("The LinearPropertySlab demo shows:")
    print("1. Comparison between linear gradients and constant approximations")
    print("2. Effects of different gradient strengths")
    print("3. Convergence behavior with sub-slab resolution")
    print("4. How atmospheric gradients affect spectral line profiles")
