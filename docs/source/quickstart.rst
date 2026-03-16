Quick Start & Examples
======================

This guide shows you how to run a basic synthesis with SolRaT. We'll use the **multi-term atom model** with a simple constant-property slab atmosphere.

Basic Stokes Profile Synthesis
------------------------------
This example demonstrates a minimal workflow for synthesizing the Stokes profiles of the He I D3 line
using a built-in pre-configured model through the public API.

.. code-block:: python

    import logging

    import numpy as np
    from yatools import logging_config

    from solrat.atom_model.model_registry import PreconfiguredModels
    from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
    from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
    from solrat.atom_model.shared.object.angles import Angles
    from solrat.atom_model.shared.object.stokes import Stokes
    from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
    from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter

    # Set up the logger
    logging_config.init(logging.INFO)

    # Get a built-in pre-configured model for the D3 transition
    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda = model.config.reference_lambda_A

    # The calculation itself needs frequency, but we will display the results in wavelength
    # Define the wavelength grid of interest
    lambda_A = np.arange(reference_lambda - 0.5, reference_lambda + 0.8, 5e-4)
    nu = lambda_A_to_frequency_hz(lambda_A)

    # Set up the observation geometry. See Fig. 5.9 in LL04 for reference
    angles = Angles(
        chi=0,
        theta=np.pi/4,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )

    # Define the atmosphere parameters
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config,
        magnetic_field_gauss=1000,
        temperature_K=5000,
    )

    # Construct a radiation tensor from the built-in height-stratified parametrization
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(h_arcsec=30)

    # Set up the initial Stokes vector: zero in this case, corresponding to a limb observation
    initial_stokes = Stokes.from_zeros(nu_sm1=nu)

    # Define the atmosphere. Here we use a MultiSlabAtmosphere with a single slab
    atmosphere = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            line_delta_tau=0.1,
            continuum_delta_tau=0.01,
            angles=angles,
            atmosphere_parameters=atmosphere_parameters,
        )
    )

    # Set up a plotter for easier visualization of results
    plotter = StokesPlotter("Stokes profiles for the He I D3 line")

    # Calculate the emerging Stokes profiles and add them to the plot
    plotter.add_stokes(
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda,
        stokes=atmosphere.forward(initial_stokes=initial_stokes),
        normalize=True,
    )

    # Show the plot
    plotter.show()

Semi-LTE synthesis of Mn I 5432 A line profiles
-----------------------------------------------
This example demonstrates a workflow for using the semi-LTE model of the Mn I 5432 A line.

.. code-block:: python

    import logging

    import numpy as np
    from yatools import logging_config

    from solrat.atom_model.model_registry import PreconfiguredModels
    from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
    from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
    from solrat.atom_model.shared.object.angles import Angles
    from solrat.atom_model.shared.object.stokes import Stokes
    from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
    from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter_IV_IpmV

    # Set up the logger
    logging_config.init(logging.INFO)

    # Get a built-in pre-configured model for the Mn I 5432 transition
    model = PreconfiguredModels.multi_term_atom_lte_MnI_5432()
    reference_lambda = model.config.reference_lambda_A

    # The calculation itself needs frequency, but we will display the results in wavelength
    # Define the wavelength grid of interest
    lambda_A = np.arange(reference_lambda + 1.5 - 0.5, reference_lambda + 1.1 + 1, 1e-3)
    nu = lambda_A_to_frequency_hz(lambda_A)

    # Set up the observation geometry. See Fig. 5.9 in LL04 for reference
    angles = Angles(chi=0, theta=0, gamma=0, chi_B=0, theta_B=0)

    # Initial Stokes vector: initialize from the Planck's BP at T=5700 K
    initial_stokes = Stokes.from_BP(nu_sm1=nu, temperature_K=5700)

    # Set up the two-slab atmosphere model
    atmosphere_Mn = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=model.RadiationTensor(),
            line_delta_tau=0.3,
            continuum_delta_tau=0.01,
            angles=angles,
            atmosphere_parameters=model.AtmosphereParameters(
                model_config=model.config,
                magnetic_field_gauss=1000,
                temperature_K=5000,
                delta_v_turbulent_cm_sm1=1000_00,
                macroscopic_velocity_cm_sm1=0,
                voigt_a=0,
            ),
        ),
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=model.RadiationTensor(),
            line_delta_tau=0.2,
            continuum_delta_tau=0.01,
            angles=angles,
            atmosphere_parameters=model.AtmosphereParameters(
                model_config=model.config,
                magnetic_field_gauss=2000,
                temperature_K=6000,
                delta_v_turbulent_cm_sm1=1000_00,
                macroscopic_velocity_cm_sm1=0,
                voigt_a=0,
            ),
        ),
    )

    # Calculate the emerging Stokes profiles
    stokes_Mn = atmosphere_Mn.forward(initial_stokes=initial_stokes)

    # Set up a plotter for easier visualization of results
    plotter = StokesPlotter_IV_IpmV("Mn I 5432 Line Stokes profiles")

    plotter.add_stokes(
        lambda_A=lambda_A,
        reference_lambda_A=1.5,
        stokes=Stokes(
            nu=stokes_Mn.nu,
            I=stokes_Mn.I,
            Q=stokes_Mn.Q,
            U=stokes_Mn.U,
            V=stokes_Mn.V,
        ),
        stokes_reference=initial_stokes,  # Use the initial Stokes I as the reference for Stokes profile scaling
        label="Mn I 5432 line",
    )

    plotter.show()


Custom Models
-------------
For customizing the models, please install SolRaT in the development mode:

.. code-block:: bash

   git clone https://github.com/yakovkinii/SolRaT.git
   cd SolRaT
   pip install -e .

Then the models can be modified by following the examples of :any:`multi_term_atom`,
:any:`multi_term_atom_legacy`, and :any:`multi_term_atom_lte` models. The models can be independent
like :any:`multi_term_atom`, or introduce slight modifications while reusing most of the other model's features
like :any:`multi_term_atom_lte`.

Next
----
For more examples, please explore the `_demos/` and `_tests/` directories in the `GitHub repository <https://github.com/yakovkinii/SolRaT/>`_.
Also, check out the full API reference starting with the :doc:`api` for detailed documentation.