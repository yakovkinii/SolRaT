import logging

import numpy as np
from matplotlib import pyplot as plt
from numpy import real
from yatools import logging_config

from src.core.physics.functions import lambda_cm_to_frequency_hz
from src.two_term_atom.atomic_data.HeI import fill_precomputed_He_I_D3_data, get_He_I_D3_data
from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.radiative_transfer_equations import RadiativeTransferCoefficients
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

    term_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()

    lambda_A = np.arange(reference_lambda_A - 1, reference_lambda_A + 1, 2e-3)
    nu = lambda_cm_to_frequency_hz(lambda_A * 1e-8)  # Hz

    atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=20000, delta_v_thermal_cm_sm1=1_000_00)

    radiation_tensor = RadiationTensor(transition_registry=transition_registry).fill_planck(T_K=5000)

    atom = TwoTermAtom(
        term_registry=term_registry,
        transition_registry=transition_registry,
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor=radiation_tensor,
        disable_r_s=False,
        disable_n=False,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(atom, root="../../")
    atom.add_all_equations()
    rho = atom.get_solution_direct()
    radiative_transfer_coefficients = RadiativeTransferCoefficients(
        atmosphere_parameters=atmosphere_parameters,
        transition_registry=transition_registry,
        nu=nu,
    )
    # logging.warning('START')
    # eta_sIpr = radiative_transfer_coefficients.precompute_eta_s(rho=rho, stokes_component_index=0)
    # logging.warning('END')
    eta_sIpr = real(radiative_transfer_coefficients.precompute_eta_s(rho=rho, stokes_component_index=0))
    eta_sI = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=0)
    eta_sVpr = real(radiative_transfer_coefficients.precompute_eta_s(rho=rho, stokes_component_index=3))
    eta_sV = radiative_transfer_coefficients.eta_s(rho=rho, stokes_component_index=3)

    plt.plot(lambda_A - reference_lambda_A, eta_sI / max(eta_sI), label=r"$\eta_s$ (Stokes $I$)")
    plt.plot(lambda_A - reference_lambda_A, eta_sIpr / max(eta_sIpr),':', label=r"$\eta_s$ pr (Stokes $I$)")
    plt.plot(lambda_A - reference_lambda_A, eta_sV / max(eta_sI), label=r"$\eta_s$ (Stokes $V$)")
    plt.plot(lambda_A - reference_lambda_A, eta_sVpr / max(eta_sIpr),":", label=r"$\eta_s$ pr (Stokes $V$)")
    plt.xlabel(r"$\Delta\lambda$ ($\AA$)")
    plt.ylabel(r"$\eta_s$ (a.u.)")
    plt.title(rf"He I D3: $\eta_s$ vs $\Delta\lambda$. $B_z = {atmosphere_parameters.magnetic_field_gauss//1000}$ kG")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
