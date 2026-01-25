import logging

import numpy as np
from numpy import real
from yatools import logging_config

from src.common.functions import lambda_cm_to_frequency_hz
from src.gui.plots.plot_stokes_profiles import StokesPlotter_IV_IpmV
from src.multi_term_atom.atomic_data.FeI import get_Fe_I_5434_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEELTE


def main():
    logging_config.init(logging.INFO)

    # Load the atomic data for He I D3
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu = get_Fe_I_5434_data()

    # The calculation itself needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda_A + 1.4, reference_lambda_A + 1.6, 1e-3)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

    see = MultiTermAtomSEELTE(
        level_registry=level_registry,
    )

    # Set up the radiative transfer equations
    rte = MultiTermAtomRTE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        nu=nu,
        precompute=False,
        j_constrained=True,
    )

    # Set up the plotter
    plotter = StokesPlotter_IV_IpmV(title="Fe I 5434: Emission coefficient vs wavelength (LTE)")

    # loop through the magnetic field values
    for Bzi in [0, 1, 2]:
        Bz = Bzi * 10000
        # Set up the atmosphere parameters
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz,
            temperature_K=7000,
            atomic_mass_amu=atomic_mass_amu,
        )

        rho = see.get_solution(atmosphere_parameters=atmosphere_parameters)

        frame_a_frame = rte.calculate_eta_rho_a(
            stokes_component_index=0,
            angles=Angles(
                chi=0,
                theta=0,
                gamma=0,
                chi_B=0,
                theta_B=0,
            ),
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
        )

        frame_a_frame_V = rte.calculate_eta_rho_a(
            stokes_component_index=3,
            angles=Angles(
                chi=0,
                theta=0,
                gamma=0,
                chi_B=0,
                theta_B=0,
            ),
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
        )

        plotter.add(
            lambda_A=lambda_A,
            stokes_I=real(frame_a_frame),
            stokes_V=real(frame_a_frame_V),
            reference_lambda_A=reference_lambda_A + 1.5,
            color="auto",
            label=rf"$B_z = {Bz/1000:.0f}$ kG",
        )

    plotter.show()


if __name__ == "__main__":
    main()
