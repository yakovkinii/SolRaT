Quick Start & Examples
======================

This guide shows you how to run a basic synthesis with SolRaT. We'll use the **multi-term atom model** with a simple constant-property slab atmosphere.

For more detailed examples, please explore the `_demos/` and `_tests/` directories in the `GitHub repository <https://github.com/yakovkinii/SolRaT/>`_.

Basic Stokes Profile Synthesis
------------------------------
This example demonstrates a minimal workflow: load atomic data for a species (here, we use a mock atom for demonstration), set up a simple atmosphere, and compute the emergent Stokes profiles.

.. code-block:: python

    import logging

    from yatools import logging_config

    from solrat.gui.plots.plot_stokes_profiles import StokesPlotter
    from solrat.multi_term_atom.atmosphere.constant_property_slab import ConstantPropertySlabAtmosphere
    from solrat.multi_term_atom.atmosphere.multi_slab_atmosphere import MultiSlabAtmosphere
    from solrat.multi_term_atom.atomic_data.HeI import create_He_I_D3_context
    from solrat.multi_term_atom.object.angles import Angles
    from solrat.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
    from solrat.multi_term_atom.object.radiation_tensor import RadiationTensor
    from solrat.multi_term_atom.object.stokes import Stokes

    # Set up prettier logs
    logging_config.init(logging.INFO)

    # Get the built-in He I D3 atom context: atomic model, transitions, relevant spectral range, etc.
    context = create_He_I_D3_context(lambda_range_A=0.75, lambda_resolution_A=1e-3)

    # Set up the observation angles
    angles = Angles(
        chi=0,
        theta=45,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )

    # Prepare the plotter
    plotter = StokesPlotter("He I D3 transition for different magnetic field values")

    # Loop through different physical parameters, magnetic field (G) in this case
    for Bz in [0, 3000, 5000]:
        # Specify the atmosphere parameters
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz, temperature_K=5000, atomic_mass_amu=context.atomic_mass_amu
        )

        # Specify the radiation tensor from tabulated anisotropic radiation at 30 arcsec above photosphere.
        radiation_tensor = RadiationTensor(context.transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=30)

        # Initial Stokes profiles are zeros (limb event)
        initial_stokes = Stokes.from_zeros(nu_sm1=context.nu)

        # Finalize the atmosphere setup, in this case a single constant property slab.
        atmosphere = MultiSlabAtmosphere(
            ConstantPropertySlabAtmosphere(
                multi_term_atom_context=context,
                radiation_tensor=radiation_tensor,
                line_delta_tau=0.1,
                continuum_delta_tau=0.01,
                angles=angles,
                atmosphere_parameters=atmosphere_parameters,
            )
        )

        # Perform SEE+RTE forward modeling loop
        stokes=atmosphere.forward(initial_stokes=initial_stokes)

        # Plot the resulting Stokes profiles
        plotter.add_stokes(
            lambda_A=context.lambda_A,
            reference_lambda_A=context.reference_lambda_A,
            stokes=stokes,
            label=f"B = {Bz} G",
            normalize=True,
        )

    # Show the plot
    plotter.show()

