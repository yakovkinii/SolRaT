import logging
from pathlib import Path

import numpy as np
from numpy import real
from yatools import logging_config

from src.common.constants import atomic_mass_unit_g, kB_erg_Km1
from src.common.functions import lambda_cm_to_frequency_hz
from src.gui.plots.plot_stokes_profiles import StokesPlotter_IV
from src.multi_term_atom.atomic_data.HeI import (
    fill_precomputed_He_I_D3_data,
    get_He_I_D3_data,
)
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE


def main():
    """
    This demo shows the calculation of stimulated emission eta_S profiles
    for the He I D3 transition under super-strong magnetic fields.
    This result is closely related to Fig. 8 in Yakovkin & Lozitsky (MNRAS, 2023)
    https://doi.org/10.1093/mnras/stad1816, where these profiles were obtained using HAZEL2.
    In the mentioned paper, the Stokes profiles are shown instead, but they should resemble eta_S for low optical depth.
    """

    logging_config.init(logging.INFO)

    # Load the atomic data for He I D3
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu = get_He_I_D3_data()

    # The calculation itself needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda_A - 2, reference_lambda_A + 2, 5e-4)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

    # Set up the statistical equilibrium equations.
    # Do not pre-compute coefficients, load them from file instead for the purpose of this demo.
    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.as_posix())

    angles = Angles(
        chi=0,
        theta=0,
        gamma=0,
        chi_B=0,
        theta_B=0,
    )

    # Set up the radiative transfer equations
    rte = MultiTermAtomRTE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        nu=nu,
    )

    # Fill the radiation tensor with anisotropic radiation field 10 arcsec from the Sun's apparent surface
    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=10)
    # radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_planck(T_K=5000)

    # Set up the plotter
    plotter = StokesPlotter_IV(title="He I D3: Emission coefficient vs wavelength")

    # loop through the magnetic field values
    for Bz in [20000, 40000, 60000, 80000, 100000]:
        # for Bz in [20000]:
        # Set up the atmosphere parameters
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz,
            temperature_K=1_000_00**2 / kB_erg_Km1 / 2 * 4 * atomic_mass_unit_g,
            atomic_mass_amu=atomic_mass_amu,
        )

        # Construct all equations for rho
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
        )

        # Solve all equations for rho
        rho = see.get_solution()

        rtc = rte.calculate_all_coefficients(atmosphere_parameters=atmosphere_parameters, rho=rho, angles=angles)
        eta_rho_sI = rtc.eta_rho_sI
        eta_rho_sV = rtc.eta_rho_sV

        plotter.add(
            lambda_A=lambda_A,
            stokes_I=real(eta_rho_sI),
            stokes_V=real(eta_rho_sV),
            reference_lambda_A=reference_lambda_A,
            color="auto",
            label=rf"$B_z = {Bz/1000:.0f}$ kG",
        )

    plotter.show()


if __name__ == "__main__":
    main()
