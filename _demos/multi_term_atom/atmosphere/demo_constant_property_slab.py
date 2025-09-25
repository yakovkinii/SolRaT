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

    import matplotlib.pyplot as plt
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


def demo_parameter_study():
    """
    Demonstrate parameter study: effect of optical depth and thermal velocity.
    """
    print("\n=== Parameter Study Demo ===")
    print("Studying effects of optical depth and thermal velocity...")

    # Create context
    context = create_he_i_d3_context(lambda_range_A=0.6, lambda_resolution_A=3e-4)
    radiation_tensor = radiation_tensor_NLTE_n_w_parametrized(context, h_arcsec=10)

    # Fixed parameters
    B_field = 2000  # Gauss

    # Study optical depth effects
    tau_values = [0.5, 1.0, 2.0, 4.0]
    thermal_velocity = 40000

    print(f"Fixed B = {B_field} G, thermal velocity = {thermal_velocity} cm/s")

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Parameter Study: Optical Depth and Thermal Velocity Effects", fontsize=14)

    # Optical depth study
    stokes_tau = []
    for tau in tau_values:
        slab = ConstantPropertySlab(
            multi_term_atom_context=context,
            radiation_tensor=radiation_tensor,
            tau=tau,
            chi=0, theta=0, gamma=0,
            magnetic_field_gauss=B_field,
            chi_B=0, theta_B=0,
            delta_v_thermal_cm_sm1=thermal_velocity,
            initial_stokes=1.0
        )
        stokes = slab.forward()
        stokes_tau.append(stokes)

    # Plot tau effects
    from src.common.functions import frequency_hz_to_lambda_A
    for i, (tau, stokes) in enumerate(zip(tau_values, stokes_tau)):
        lambda_A = frequency_hz_to_lambda_A(stokes.nu)
        delta_lambda = lambda_A - context.reference_lambda_A
        axs[0, 0].plot(delta_lambda, stokes.I, label=f'τ = {tau}')
        axs[0, 1].plot(delta_lambda, stokes.V, label=f'τ = {tau}')

    axs[0, 0].set_ylabel('Stokes I')
    axs[0, 0].set_title('Effect of Optical Depth')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].set_ylabel('Stokes V')
    axs[0, 1].set_title('Effect of Optical Depth')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Thermal velocity study
    v_thermal_values = [20000, 40000, 60000, 80000]  # cm/s
    tau_fixed = 1.5

    stokes_vth = []
    for v_th in v_thermal_values:
        slab = ConstantPropertySlab(
            multi_term_atom_context=context,
            radiation_tensor=radiation_tensor,
            tau=tau_fixed,
            chi=0, theta=0, gamma=0,
            magnetic_field_gauss=B_field,
            chi_B=0, theta_B=0,
            delta_v_thermal_cm_sm1=v_th,
            initial_stokes=1.0
        )
        stokes = slab.forward()
        stokes_vth.append(stokes)

    # Plot thermal velocity effects
    for i, (v_th, stokes) in enumerate(zip(v_thermal_values, stokes_vth)):
        lambda_A = frequency_hz_to_lambda_A(stokes.nu)
        delta_lambda = lambda_A - context.reference_lambda_A
        axs[1, 0].plot(delta_lambda, stokes.I, label=f'v_th = {v_th/1000:.0f} km/s')
        axs[1, 1].plot(delta_lambda, stokes.V, label=f'v_th = {v_th/1000:.0f} km/s')

    axs[1, 0].set_ylabel('Stokes I')
    axs[1, 0].set_xlabel('Δλ [Å]')
    axs[1, 0].set_title('Effect of Thermal Velocity')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].set_ylabel('Stokes V')
    axs[1, 1].set_xlabel('Δλ [Å]')
    axs[1, 1].set_title('Effect of Thermal Velocity')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.show()


if __name__ == "__main__":
    demo_constant_property_slab()
    demo_parameter_study()

    print("\n=== Demo Complete ===")
    print("The ConstantPropertySlab demo shows:")
    print("1. Basic usage for different magnetic field strengths")
    print("2. Parameter studies for optical depth and thermal velocity")
    print("3. How magnetic fields produce Zeeman splitting and circular polarization")
