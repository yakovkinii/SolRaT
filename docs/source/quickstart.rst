Quick Start & Examples
======================

This guide shows you how to run a basic synthesis with SolRaT. We'll use the **multi-term atom model** with a simple constant-property slab atmosphere.

For more detailed examples, please explore the `_demos/` and `_tests/` directories in the `GitHub repository <https://github.com/yakovkinii/SolRaT/>`_.

Basic Stokes Profile Synthesis
------------------------------
This example demonstrates a minimal workflow: load atomic data for a species (here, we use a mock atom for demonstration), set up a simple atmosphere, and compute the emergent Stokes profiles.

.. code-block:: python

    import logging

    import numpy as np
    from yatools import logging_config

    from solrat.atom_model.model_registry import PreconfiguredModels
    from solrat.atom_model.shared.common_api.constant_property_slab import (
        ConstantPropertySlabAtmosphere,
    )
    from solrat.atom_model.shared.common_api.multi_slab_atmosphere import (
        MultiSlabAtmosphere,
    )
    from solrat.atom_model.shared.object.angles import Angles
    from solrat.atom_model.shared.object.stokes import Stokes
    from solrat.atom_model.shared.utility.functions import lambda_A_to_frequency_hz
    from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter

    logging_config.init(logging.INFO)

    # Get a built-in pre-computed model for the D3 transition
    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda = model.config.reference_lambda_A

    # The calculation itself needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda - 2, reference_lambda + 2, 5e-4)
    nu = lambda_A_to_frequency_hz(lambda_A)

    angles = Angles(
        chi=0,
        theta=45,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )

    plotter = StokesPlotter("He I D3 transition for different magnetic field values")

    for Bz in [0, 3000, 5000]:
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=Bz,
            temperature_K=5000,
        )

        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_parametrized(
            h_arcsec=30,
        )

        initial_stokes = Stokes.from_zeros(nu_sm1=nu)
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

        plotter.add_stokes(
            lambda_A=lambda_A,
            reference_lambda_A=reference_lambda,
            stokes=atmosphere.forward(initial_stokes=initial_stokes),
            label=f"B = {Bz} G",
            normalize=True,
        )

    plotter.show()



