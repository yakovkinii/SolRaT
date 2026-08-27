Quick Start & Examples
======================

This guide shows you how to run a basic synthesis with SolRaT. We'll use the **multi-term atom model** with a simple constant-property slab atmosphere.

Basic Stokes Profile Synthesis
------------------------------
This example demonstrates a minimal workflow for synthesizing the Stokes profiles of the He I D3 line
using a built-in pre-configured model through the public API.

.. code-block:: python

    import numpy as np

    from solrat.atom_model.model_registry import PreconfiguredModels
    from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
    from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
    from solrat.atom_model.shared.object.angles import Angles
    from solrat.atom_model.shared.object.stokes import Stokes
    from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
    from solrat.atom_model.shared.utility.log_setup import setup_logging
    from solrat.atom_model.shared.utility.plot_stokes_profiles import StokesPlotter


    def main():
        """
        This example demonstrates a minimal workflow for synthesizing the Stokes profiles of the He I D3 line
        using a built-in pre-configured model through the public API.

        To run this demo, SolRaT needs to be installed.
        The easiest way is to install SolRaT using "pip install solrat" command.

        You can also clone the repository and run _demos/start_here/demo_basic_stokes_profile_synthesis.py directly.
        The latter is recommended if you plan to extend available models or creating your own.
        """

        # Set up the logger. Use setup_logging(logging.DEBUG) for a verbose logging
        setup_logging()

        # Get a built-in pre-configured model for the D3 transition
        # You can browse PreconfiguredModels to see all available pre-configured transfer models.
        # Additionally, you can configure a transfer model for your own atom.
        # The latter can be easily done by direct analogy to pre-configured models.
        model = PreconfiguredModels.multi_term_atom_HeID3()
        reference_lambda_A_air = model.config.reference_lambda_A_air

        # The calculation itself needs frequency, but we will display the results in air wavelength.
        # Therefore, it is convenient to derive the frequencies from the wavelengths range of interest.
        # Note that SolRaT scales gracefully with the number of frequency points,
        # so feel free to request a high spectral resolution.
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.3,
            upper_wavelength_A=reference_lambda_A_air + 0.6,
            step_A=5e-3,
        )

        # Set up the observation geometry. See Fig. 5.9 in LL04 for reference
        # In short:
        # chi, theta, and gamma are the Euler angles for the Omega (LOS) vector.
        # chi_B and theta_B are the Euler angles for the magnetic field B vector.
        angles = Angles(
            chi=0,
            theta=np.pi / 4,
            gamma=0,
            chi_B=0,
            theta_B=0,
        )

        # Define the atmosphere parameters
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=1000,
            temperature_K=5000,
            delta_v_turbulent_cm_sm1=0,  # microturbulent velocity [cm/s]
            macroscopic_velocity_cm_sm1=0,  # Doppler shift, typically used when multiple emission components are modeled
            voigt_a=0,  # Voigt damping coefficient
        )

        # Construct a radiation tensor from the Allen solar radiation-field machinery.
        # Here we use the radiation tensor at a height of 30 arcsec above the photosphere.
        # Another option is .fill_planck(self, temperature_K) - fill using LTE Planck distribution
        radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=30)

        # Set up the initial Stokes vector: zero in this case, corresponding to a limb observation
        # Another option is .from_BP(nu_sm1, temperature_K) - fill using LTE Planck distribution
        initial_stokes = Stokes.from_zeros(nu_sm1=nu)

        # Define the atmosphere. Here we use a MultiSlabAtmosphere with a single slab
        atmosphere = MultiSlabAtmosphere(
            ConstantPropertySlabAtmosphere(
                model=model,
                radiation_tensor=radiation_tensor,
                line_delta_tau=0.1,  # optical depth of the slab in the line-center frequency
                continuum_delta_tau=0.001,  # optical depth in the far continuum
                angles=angles,
                atmosphere_parameters=atmosphere_parameters,
            ),
            # Subsequent slabs can be added here
        )

        # Set up a plotter for easier visualization of results
        plotter = StokesPlotter("Stokes profiles for the He I D3 line", reference_lambda_A_air=reference_lambda_A_air)

        # Calculate the emerging Stokes profiles and add them to the plot
        emerging_stokes = atmosphere.forward(initial_stokes=initial_stokes)

        # Plot the emerging Stokes profiles
        plotter.add_stokes(
            nu=nu,
            stokes=emerging_stokes,
            norm=plotter.Norm.MAX_I,  # Normalize using maximum I intensity. Other options: BY_REFERENCE, MAX_IpV_ImV, NONE
            # stokes_reference=initial_stokes # for on-disk observations, used when norm is BY_REFERENCE.
            label="$B=1000$ G",
        )

        # Show the plot
        plotter.show()


    if __name__ == "__main__":
        main()

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
