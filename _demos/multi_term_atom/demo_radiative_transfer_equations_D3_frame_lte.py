import logging
from pathlib import Path

import numpy as np
from numpy import real, imag
from yatools import logging_config

from src.common.functions import lambda_cm_to_frequency_hz
from src.gui.plots.plot_stokes_profiles import StokesPlotter_IV
from src.multi_term_atom.atomic_data.HeI import fill_precomputed_He_I_D3_data, get_He_I_D3_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE, MultiTermAtomSEELTE


def main():
    logging_config.init(logging.INFO)

    # Load the atomic data for He I D3
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()

    # The calculation itself needs frequency, but we will display the results in wavelength
    lambda_A = np.arange(reference_lambda_A - 2, reference_lambda_A + 2, 1e-3)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)

    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.as_posix())

    seelte = MultiTermAtomSEELTE(
        level_registry=level_registry,
        atomic_mass_amu=4,  # He
    )


    # Set up the radiative transfer equations
    # Angles input is optional. But since we know the angles in advance,
    # we provide them here to speed up the calculation.
    rte = MultiTermAtomRTE(
        level_registry=level_registry,
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
        precompute=False,
    )



    # Fill the radiation tensor with anisotropic radiation field 10 arcsec from the Sun's apparent surface
    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=10)
    # radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_planck(T_K=5000)

    # Set up the plotter
    plotter = StokesPlotter_IV(title="He I D3: Emission coefficient vs wavelength")

    # loop through the magnetic field values
    Bz = 20000
    for lte in [False, True]:
        # Set up the atmosphere parameters
        atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=Bz,
            delta_v_thermal_cm_sm1=10_000_00,
        )
        if lte:
            rho = seelte.get_solution(atmosphere_parameters=atmosphere_parameters)
        else:
            # Construct all equations for rho
            see.fill_all_equations(
                atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
            )

            # Solve all equations for rho
            rho = see.get_solution()

        logging.warning('START')
        logging.warning('START')
        logging.warning('START')
        logging.warning('START')
        frame_a_frame = rte.frame_a_frame(
            stokes_component_index=0,
            angles=Angles(
                chi=0,
                theta=0,
                gamma=0,
                chi_B=0,
                theta_B=0,
            ),
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
        )
        logging.warning('END')
        logging.warning('END')
        logging.warning('END')
        logging.warning('END')
        logging.info(f"frame_a_frame={frame_a_frame}")
        frame_a_frame_V = rte.frame_a_frame(
            stokes_component_index=3,
            angles=Angles(
                chi=0,
                theta=0,
                gamma=0,
                chi_B=0,
                theta_B=0,
            ),
            magnetic_field_gauss=atmosphere_parameters.magnetic_field_gauss,
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
        )
        # get RT coefficients. They are complex: eta = real(eta_rho), rho = imag(eta_rho)
        # eta_rho_aI, eta_rho_aQ, eta_rho_aU, eta_rho_aV = rte.eta_rho_a(
        #     atmosphere_parameters=atmosphere_parameters,
        #     # angles=...,  # We could provide angles here if they were dynamic
        #     rho=rho,
        # )

        # rtc = rte.compute_all_coefficients(
        #     atmosphere_parameters=atmosphere_parameters,
        #     rho=rho,
        # )
        # eta_rho_sI = rtc.eta_rho_sI
        # eta_rho_sV = rtc.eta_rho_sV

        plotter.add(
            lambda_A=lambda_A,
            stokes_I=real(frame_a_frame),
            stokes_V=real(frame_a_frame_V),
            reference_lambda_A=reference_lambda_A,
            color="auto",
            label=rf"$B_z = {Bz/1000:.0f}$ kG, {'LTE' if lte else 'NLTE'}",
        )
        # logging.info((np.abs(eta_rho_aI-frame_a_frame)).max())
        # logging.info((np.abs(eta_rho_aI-frame_a_frame)/np.abs(eta_rho_aI)).max())
    plotter.show()


if __name__ == "__main__":
    main()
