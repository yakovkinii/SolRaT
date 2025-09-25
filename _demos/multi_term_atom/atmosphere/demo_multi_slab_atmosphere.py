"""
Demo for MultiSlabAtmosphere: Complex atmospheric structures with multiple components.

This demo shows how to use the MultiSlabAtmosphere class to model:
1. Two-component atmospheres (filling factors)
2. Stratified atmospheres (sequential layers)
3. Mixed scenarios combining both approaches
"""

import logging
import numpy as np
from yatools import logging_config

from src.multi_term_atom.object.multi_term_atom_context import create_he_i_d3_context
from src.multi_term_atom.atmosphere.constant_property_slab import ConstantPropertySlab
from src.multi_term_atom.atmosphere.linear_property_slab import LinearPropertySlab
from src.multi_term_atom.atmosphere.multi_slab_atmosphere import (
    MultiSlabAtmosphere,
    create_two_component_atmosphere,
    create_stratified_atmosphere
)
from src.multi_term_atom.atmosphere.utils import radiation_tensor_NLTE_n_w_parametrized, plot_stokes_comparison


def demo_two_component_atmosphere():
    """
    Demonstrate two-component atmosphere modeling using filling factors.
    """
    logging_config.init(logging.INFO)

    print("=== Two-Component Atmosphere Demo ===")
    print("Modeling mixed quiet + active regions using filling factors...")

    # Create atomic context
    context = create_he_i_d3_context(lambda_range_A=0.8, lambda_resolution_A=5e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Test different filling factor combinations
    scenarios = [
        {"ff1": 1.0, "ff2": 0.0, "label": "100% Quiet"},
        {"ff1": 0.7, "ff2": 0.3, "label": "70% Quiet + 30% Active"},
        {"ff1": 0.5, "ff2": 0.5, "label": "50% Quiet + 50% Active"},
        {"ff1": 0.3, "ff2": 0.7, "label": "30% Quiet + 70% Active"},
        {"ff1": 0.0, "ff2": 1.0, "label": "100% Active"},
    ]

    stokes_results = []
    labels = []

    for scenario in scenarios:
        print(f"Computing: {scenario['label']}")

        # Create two-component atmosphere
        atmosphere = create_two_component_atmosphere(
            context, radiation_tensor,
            tau1=1.2,      # Quiet region optical depth
            B1=800,        # Quiet region field (Gauss)
            tau2=1.8,      # Active region optical depth
            B2=3500,       # Active region field (Gauss)
            filling_factor1=scenario["ff1"],
            filling_factor2=scenario["ff2"]
        )

        # Solve using filling factors
        stokes = atmosphere.forward_filling_factors(initial_stokes=1.0)
        stokes_results.append(stokes)
        labels.append(scenario["label"])

    # Plot comparison
    plot_stokes_comparison(
        stokes_results, labels, context.reference_lambda_A,
        title="Two-Component Atmosphere: Quiet + Active Regions"
    )

    return stokes_results


def demo_stratified_atmosphere():
    """
    Demonstrate stratified atmosphere with sequential layers.
    """
    print("\n=== Stratified Atmosphere Demo ===")
    print("Modeling layered atmosphere with increasing magnetic field...")

    # Create context
    context = create_he_i_d3_context(lambda_range_A=0.8, lambda_resolution_A=5e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Compare different numbers of layers
    layer_numbers = [1, 3, 5, 8]
    stokes_results = []
    labels = []

    for n_layers in layer_numbers:
        print(f"Computing {n_layers} layer atmosphere...")

        # Create stratified atmosphere
        atmosphere = create_stratified_atmosphere(
            context, radiation_tensor, n_layers=n_layers
        )

        print(f"  Total optical depth: {atmosphere.get_total_optical_depth():.2f}")

        # Solve sequentially
        stokes = atmosphere.forward_sequential(initial_stokes=1.0)
        stokes_results.append(stokes)
        labels.append(f"{n_layers} layers")

    # Plot comparison
    plot_stokes_comparison(
        stokes_results, labels, context.reference_lambda_A,
        title="Stratified Atmosphere: Effect of Layer Number"
    )

    return stokes_results


def demo_custom_multi_slab():
    """
    Demonstrate custom multi-slab configurations.
    """
    print("\n=== Custom Multi-Slab Demo ===")
    print("Creating custom atmosphere with mixed constant and gradient slabs...")

    # Create context
    context = create_he_i_d3_context(lambda_range_A=0.6, lambda_resolution_A=3e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Create custom atmosphere
    atmosphere = MultiSlabAtmosphere()

    # Layer 1: Constant weak field region (chromosphere)
    chromosphere = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=0.8,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=200,
        delta_v_thermal_cm_sm1=35000,
    )
    atmosphere.add_slab_sequential(chromosphere)

    # Layer 2: Linear gradient transition region
    transition = LinearPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau_total=0.6,
        magnetic_field_gauss_top=200,
        magnetic_field_gauss_bottom=2000,
        delta_v_thermal_cm_sm1_top=35000,
        delta_v_thermal_cm_sm1_bottom=45000,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        n_sub_slabs=8,
    )
    atmosphere.add_slab_sequential(transition)

    # Layer 3: Constant strong field region (photosphere)
    photosphere = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=1.2,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=2000,
        delta_v_thermal_cm_sm1=45000,
    )
    atmosphere.add_slab_sequential(photosphere)

    print(f"Custom atmosphere: {len(atmosphere.slabs)} layers")
    print(f"Total optical depth: {atmosphere.get_total_optical_depth():.2f}")

    # Compute solution
    print("Computing custom atmosphere solution...")
    stokes_custom = atmosphere.forward_sequential(initial_stokes=1.0)

    # Compare with simple constant slab
    B_avg = (200 + 2000) / 2  # Average field
    v_avg = (35000 + 45000) / 2  # Average velocity
    tau_total = atmosphere.get_total_optical_depth()

    simple_slab = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=tau_total,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=B_avg,
        delta_v_thermal_cm_sm1=v_avg,
        initial_stokes=1.0
    )

    print("Computing equivalent constant slab...")
    stokes_simple = simple_slab.forward()

    # Plot comparison
    plot_stokes_comparison(
        [stokes_custom, stokes_simple],
        ["Custom multi-layer", "Equivalent constant"],
        context.reference_lambda_A,
        title="Custom Multi-Slab vs Simple Constant Slab"
    )

    return stokes_custom, stokes_simple


def demo_mixed_filling_factors():
    """
    Demonstrate mixed scenarios with both filling factors and stratification.
    """
    print("\n=== Mixed Filling Factors Demo ===")
    print("Combining stratified and filling factor approaches...")

    # Create context
    context = create_he_i_d3_context(lambda_range_A=0.6, lambda_resolution_A=4e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Scenario: Network bright points + quiet background
    atmosphere = MultiSlabAtmosphere()

    # Component 1: Quiet background (80% filling)
    quiet_atmosphere = MultiSlabAtmosphere()

    # Quiet: weak field chromosphere + photosphere
    quiet_chrom = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=0.5,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=100,
        delta_v_thermal_cm_sm1=30000,
    )
    quiet_atmosphere.add_slab_sequential(quiet_chrom)

    quiet_phot = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=1.0,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=500,
        delta_v_thermal_cm_sm1=40000,
    )
    quiet_atmosphere.add_slab_sequential(quiet_phot)

    # Component 2: Network bright points (20% filling)
    network_atmosphere = MultiSlabAtmosphere()

    # Network: strong field with gradient
    network_grad = LinearPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau_total=1.8,
        magnetic_field_gauss_top=1000,
        magnetic_field_gauss_bottom=4000,
        delta_v_thermal_cm_sm1_top=40000,
        delta_v_thermal_cm_sm1_bottom=55000,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        n_sub_slabs=10,
    )
    network_atmosphere.add_slab_sequential(network_grad)

    # Solve each component
    print("Computing quiet background...")
    stokes_quiet = quiet_atmosphere.forward_sequential(initial_stokes=1.0)

    print("Computing network bright points...")
    stokes_network = network_atmosphere.forward_sequential(initial_stokes=1.0)

    # Mix with filling factors
    ff_quiet = 0.8
    ff_network = 0.2

    print(f"Mixing: {ff_quiet:.1%} quiet + {ff_network:.1%} network")

    # Manual mixing (equivalent to MultiSlabAtmosphere with filling factors)
    from src.multi_term_atom.object.stokes import Stokes
    stokes_mixed = Stokes(
        nu=stokes_quiet.nu,
        I=ff_quiet * stokes_quiet.I + ff_network * stokes_network.I,
        Q=ff_quiet * stokes_quiet.Q + ff_network * stokes_network.Q,
        U=ff_quiet * stokes_quiet.U + ff_network * stokes_network.U,
        V=ff_quiet * stokes_quiet.V + ff_network * stokes_network.V,
    )

    # Plot all components
    plot_stokes_comparison(
        [stokes_quiet, stokes_network, stokes_mixed],
        ["Quiet background", "Network bright points", "Mixed (80%/20%)"],
        context.reference_lambda_A,
        title="Mixed Atmosphere: Quiet + Network Bright Points"
    )

    return stokes_quiet, stokes_network, stokes_mixed


if __name__ == "__main__":
    # Run all demos
    print("Starting MultiSlabAtmosphere comprehensive demo...\n")

    demo_two_component_atmosphere()
    demo_stratified_atmosphere()
    demo_custom_multi_slab()
    demo_mixed_filling_factors()

    print("\n=== Demo Complete ===")
    print("The MultiSlabAtmosphere demo shows:")
    print("1. Two-component atmospheres using filling factors")
    print("2. Stratified atmospheres with sequential layers")
    print("3. Custom combinations of constant and gradient slabs")
    print("4. Mixed scenarios combining different approaches")
    print("5. How complex atmospheric structures affect He I D3 profiles")
