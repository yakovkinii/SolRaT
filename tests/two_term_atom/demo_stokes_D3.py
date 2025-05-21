import logging
from pathlib import Path

import numpy as np
from numpy import real
from yatools import logging_config

from src.core.physics.functions import lambda_cm_to_frequency_hz
from src.core.ui.plots.plot_stokes_profiles import StokesPlotter_IV, StokesPlotter
from src.two_term_atom.atomic_data.HeI import fill_precomputed_He_I_D3_data, get_He_I_D3_data
from src.two_term_atom.object.angles import Angles
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import TwoTermAtomRTE
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtomSEE


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
    term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()

    # The calculation itself needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda_A - 2, reference_lambda_A + 2, 5e-4)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

    # Set up the statistical equilibrium equations.
    # Do not pre-compute coefficients, load them from file instead for the purpose of this demo.
    see = TwoTermAtomSEE(
        term_registry=term_registry,
        transition_registry=transition_registry,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.as_posix())

    # Set up the radiative transfer equations
    # Angles input is optional. But since we know the angles in advance,
    # we provide them here to speed up the calculation.
    rte = TwoTermAtomRTE(
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
        # magnetic_field_gauss=...,  # We will vary the magnetic field in this demo, so we cannot provide it here
        # rho=...,  # Rho is dependent on the varying magnetic field, so we cannot provide it here
    )

    # Fill the radiation tensor with anisotropic radiation field 10 arcsec from the Sun's apparent surface
    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=10)
    # radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_planck(T_K=5000)

    # Set up the plotter
    plotter = StokesPlotter(title="He I D3: Stokes profiles")

    Bz = 20000
    # Set up the atmosphere parameters
    atmosphere_parameters = AtmosphereParameters(
        magnetic_field_gauss=Bz,
        delta_v_thermal_cm_sm1=1_000_00,
    )

    # Construct all equations for rho
    see.add_all_equations(
        atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
    )

    # Solve all equations for rho
    rho = see.get_solution_direct()

    # get RT coefficients. They are complex: eta = real(eta_rho), rho = imag(eta_rho)
    rtc = rte.compute_all_coefficients(
        atmosphere_parameters=atmosphere_parameters,
        rho=rho,
    )

    # (6.85)
    stokes = np.zeros((len(rtc.eta_rho_aI), 4, 1), dtype=np.float64)
    K = rtc.K()
    epsilon = rtc.epsilon()

    h = 100000000 # cm
    for step in range(100000):
        stokes += h*(K @ stokes + epsilon)
        if step % 10000 == 0:
            plotter.add(
                lambda_A=lambda_A,
                stokes_I=real(stokes[:, 0, 0]),
                stokes_Q=real(stokes[:, 1, 0]),
                stokes_U=real(stokes[:, 2, 0]),
                stokes_V=real(stokes[:, 3, 0]),
                reference_lambda_A=reference_lambda_A,
                color="auto",
                label=rf"z={step * h/100000:.0f} km",
            )

    plotter.show()


if __name__ == "__main__":
    main()
