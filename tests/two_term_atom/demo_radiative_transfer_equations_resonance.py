import logging

import numpy as np
from matplotlib import pyplot as plt
from yatools import logging_config

from src.two_term_atom.atomic_data.mock import get_mock_atom_data
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients, Angles
from src.two_term_atom.statistical_equilibrium_equations import TwoTermAtom


def main():
    """
    This demo shows the calculation of the stimulated emission eta_S profiles
    in the case of a two-term atom under anisotropic irradiation.
    The inclined POV results in non-zero Stokes I, Q, U, and V parameters
    even when no magnetic field is present.
    The results are compared with the analytical solution.
    Reference: (10.127)
    """
    logging_config.init(logging.INFO)

    term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_mock_atom_data()
    nu = np.arange(reference_nu_sm1 - 1e11, reference_nu_sm1 + 1e11, 1e8)  # Hz

    atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=5_000_00)
    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_NLTE_w(h_arcsec=30)

    atom = TwoTermAtom(
        term_registry=term_registry,
        transition_registry=transition_registry,
        # atmosphere_parameters=atmosphere_parameters,
        # radiation_tensor=radiation_tensor,
        disable_r_s=True,
        disable_n=True,
    )

    atom.add_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor=radiation_tensor,
    )
    rho = atom.get_solution_direct()

    radiative_transfer_coefficients = RadiativeTransferCoefficients(
        term_registry=term_registry,
        # atmosphere_parameters=atmosphere_parameters,
        transition_registry=transition_registry,
        nu=nu,
        # theta=np.pi / 8,
        # gamma=np.pi / 8,
        # chi=np.pi / 8,
    )
    angles = Angles(
        chi=np.pi / 7,
        theta=np.pi/7,
        gamma=np.pi/7,
        chi_B=np.pi/5,
        theta_B=np.pi/5,
    )
    eta_sI, eta_sQ, eta_sU, eta_sV = radiative_transfer_coefficients.eta_rho_s(rho=rho, atmosphere_parameters=atmosphere_parameters, angles=angles)
    eta_sI_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=0, atmosphere_parameters=atmosphere_parameters, angles=angles)
    # eta_sQ = radiative_transfer_coefficients.eta_rho_s(rho=rho, stokes_component_index=1)
    eta_sQ_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=1, atmosphere_parameters=atmosphere_parameters, angles=angles)
    # eta_sU = radiative_transfer_coefficients.eta_rho_s(rho=rho, stokes_component_index=2)
    eta_sU_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=2, atmosphere_parameters=atmosphere_parameters, angles=angles)
    # eta_sV = radiative_transfer_coefficients.eta_rho_s(rho=rho, stokes_component_index=3)
    eta_sV_analytic = radiative_transfer_coefficients.eta_s_no_field(rho=rho, stokes_component_index=3, atmosphere_parameters=atmosphere_parameters, angles=angles)

    plt.plot(nu, eta_sI, "g-", label=r"$\eta_s^I$")
    plt.plot(nu, eta_sI_analytic, "k:", linewidth=2, label=r"$\eta_s^I$ (analytical solution)")
    plt.plot(nu, eta_sQ, "r-", label=r"$\eta_s^Q$")
    plt.plot(nu, eta_sQ_analytic, "k:", label=r"$\eta_s^Q$ (analytical solution)")
    plt.plot(nu, eta_sU, "y-", label=r"$\eta_s^U$")
    plt.plot(nu, eta_sU_analytic, "k:", label=r"$\eta_s^U$ (analytical solution)")
    plt.plot(nu, eta_sV, "b-", label=r"$\eta_s^V$")
    plt.plot(nu, eta_sV_analytic, "k:", label=r"$\eta_s^V$ (analytical solution)")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$\eta_s$")
    plt.title(r"$\eta_s$ vs Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
