import logging

import numpy as np
from matplotlib import pyplot as plt
from numpy import real
from yatools import logging_config

from src.core.physics.functions import lambda_cm_to_frequency_hz
from src.core.ui.plots.plot_stokes_profiles import StokesPlotterTwoPanel
from src.two_term_atom.atomic_data.HeI import fill_precomputed_He_I_D3_data, get_He_I_D3_data
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients, Angles
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom


def main():
    """
    This demo shows the calculation of stimulated emission eta_S profiles
    for the He I D3 transition under super-strong magnetic field of 20 kG.
    This result is closely related to Fig. 8 in Yakovkin & Lozitsky (MNRAS, 2023) https://doi.org/10.1093/mnras/stad1816
    (in the latter, the Stokes profiles are shown instead, which should resemble eta_S for low optical depth).
    This demo takes ~ 50 seconds to run.
    """

    logging_config.init(logging.INFO)

    # Load the atomic data for He I D3
    term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()

    # Calculation needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda_A - 2, reference_lambda_A + 2, 1e-3)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

    # Set up the atom (i.e. the statistical equilibrium equations).
    # Do not pre-compute coefficients, load them from file instead to save some time for this demo.
    atom = TwoTermAtom(
        term_registry=term_registry,
        transition_registry=transition_registry,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(atom, root="../../")

    # Set up RTE
    # angles input is optional. But since we know the angles beforehand,
    # it will pre-compute the coefficients for the angles, significantly speeding things up
    radiative_transfer_coefficients = RadiativeTransferCoefficients(
        term_registry=term_registry,
        transition_registry=transition_registry,
        nu=nu,
        angles=Angles(
            chi=0,
            theta=0,
            gamma=0,
            chi_B=0,
            theta_B=0,
        ),
        # magnetic_field_gauss=...,  # Not fixed, so will provide it later
        # rho=...,  # Not fixed, so will provide it later
    )

    # Fill the radiation tensor with anisotropic radiation field 30 arcsec away from the Sun
    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_w(h_arcsec=30)

    # Set up the plotter
    plotter = StokesPlotterTwoPanel(title=rf"He I D3: $\eta_s$ vs $\Delta\lambda$.")

    # loop through the magnetic field values
    for Bz in [20000, 40000, 60000, 80000, 100000]:
        # Set up the atmosphere parameters
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz,
            delta_v_thermal_cm_sm1=1_000_00,
        )

        # Construct all equations for rho
        atom.add_all_equations(atmosphere_parameters=atmosphere_parameters, radiation_tensor=radiation_tensor)

        # Solve all equations for rho
        rho = atom.get_solution_direct()

        # get RT coefficients. They are complex: eta = real(eta_rho), rho = imag(eta_rho)
        eta_rho_sI, eta_rho_sQ, eta_rho_sU, eta_rho_sV = radiative_transfer_coefficients.eta_rho_s(
            atmosphere_parameters=atmosphere_parameters,
            # angles=...,  # We could provide angles here if they vary from run to run
            rho=rho,
        )
        eta_sI = real(eta_rho_sI)
        eta_sQ = real(eta_rho_sQ)
        eta_sU = real(eta_rho_sU)
        eta_sV = real(eta_rho_sV)

        plotter.add(
            lambda_A=lambda_A,
            stokes_I=eta_sI,
            stokes_Q=eta_sQ,
            stokes_U=eta_sU,
            stokes_V=eta_sV,
            reference_lambda_A=reference_lambda_A,
            color="auto",
            label=rf"$B_z = {Bz/1000:.0f}$ kG",
        )

    plotter.show()


if __name__ == "__main__":
    main()
