import logging

from yatools import logging_config

from src.gui.plots.plot_stokes_profiles import StokesPlotter
from src.multi_term_atom.atmosphere.constant_property_slab import (
    ConstantPropertySlabAtmosphere,
)
from src.multi_term_atom.atmosphere.multi_slab_atmosphere import MultiSlabAtmosphere
from src.multi_term_atom.atomic_data.HeI import create_He_I_D3_context
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes


def main():
    """
    This demo shows the calculation of  He I D3 transition under extremely strong magnetic fields.
    This result is somewhat related to Fig. 8 in Yakovkin & Lozitsky (MNRAS, 2023)
    https://doi.org/10.1093/mnras/stad1816, where these profiles were obtained using HAZEL2.
    """

    logging_config.init(logging.INFO)

    context = create_He_I_D3_context(lambda_range_A=1, lambda_resolution_A=1e-3)

    angles = Angles(
        chi=0,
        theta=45,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )

    plotter = StokesPlotter()

    for Bz in [0, 5000, 10000]:
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz, temperature_K=5000, atomic_mass_amu=context.atomic_mass_amu
        )

        radiation_tensor = RadiationTensor(context.transition_registry).fill_NLTE_n_w_parametrized(
            h_arcsec=30,
        )

        initial_stokes = Stokes.from_zeros(nu_sm1=context.nu)
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

        plotter.add_stokes(
            lambda_A=context.lambda_A,
            reference_lambda_A=context.reference_lambda_A,
            stokes=atmosphere.forward(initial_stokes=initial_stokes),
            # stokes_reference=initial_stokes,
            label=f"B = {Bz} G",
        )

    plotter.show()


if __name__ == "__main__":
    main()
