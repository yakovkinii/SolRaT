import logging

import numpy as np
from numpy import real
from yatools import logging_config

from src.common.functions import get_planck_BP
from src.gui.plots.plot_stokes_profiles import StokesPlotter
from src.multi_term_atom.atmosphere.constant_property_slab import (
    ConstantPropertySlabAtmosphere,
)
from src.multi_term_atom.atomic_data.HeI import create_He_I_D3_context
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes


def main():
    """
    This demo shows how the DELO solver works against the different more primitive finite difference methods.
    """
    logging_config.init(logging.INFO)

    context = create_He_I_D3_context(lambda_range_A=1, lambda_resolution_A=1e-3)

    angles = Angles(
        chi=30,
        theta=40,
        gamma=50,
        chi_B=10,
        theta_B=20,
    )

    plotter = StokesPlotter()

    atmosphere_parameters = AtmosphereParameters(
        magnetic_field_gauss=10000, temperature_K=6000, atomic_mass_amu=context.atomic_mass_amu
    )

    radiation_tensor = RadiationTensor(context.transition_registry).fill_NLTE_n_w_parametrized(
        h_arcsec=30,
    )

    initial_stokes = Stokes.from_BP(nu_sm1=context.nu, temperature_K=5700)

    line_delta_tau = 0.2
    continuum_delta_tau = 0.001
    atmosphere = ConstantPropertySlabAtmosphere(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        line_delta_tau=line_delta_tau,
        continuum_delta_tau=continuum_delta_tau,
        angles=angles,
        atmosphere_parameters=atmosphere_parameters,
    )

    plotter.add_stokes(
        lambda_A=context.lambda_A,
        reference_lambda_A=context.reference_lambda_A,
        stokes=atmosphere.forward(initial_stokes=initial_stokes),
        stokes_reference=initial_stokes,
        label="DELO",
    )

    # Reuse the same RTC using different methods and reconstruct the K and epsilon
    rtc = atmosphere.rtc

    #  dStokes/dtau_line = K_tau_line * Stokes - epsilon_tau_line + Stokes / eta_LC - BP(T) eI / eta_LC

    K_tau_line = rtc.K_tau()  # [Nν, 4, 4]
    epsilon_tau_line = rtc.epsilon_tau()[:, :, 0]  # [Nν, 4]

    eta_LC = line_delta_tau / continuum_delta_tau

    # Add continuum
    K_tau_line[:, 0, 0] += 1 / eta_LC
    K_tau_line[:, 1, 1] += 1 / eta_LC
    K_tau_line[:, 2, 2] += 1 / eta_LC
    K_tau_line[:, 3, 3] += 1 / eta_LC
    epsilon_tau_line[:, 0] += get_planck_BP(nu_sm1=context.nu, T_K=atmosphere_parameters.temperature_K) / eta_LC

    # Rename to be explicit
    K_tau = K_tau_line
    epsilon_tau = epsilon_tau_line

    # Construct initial conditions:
    stokes = np.zeros((len(context.nu), 4, 1), dtype=np.float64)
    stokes[:, 0, 0] = initial_stokes.I
    stokes[:, 1, 0] = initial_stokes.Q
    stokes[:, 2, 0] = initial_stokes.U
    stokes[:, 3, 0] = initial_stokes.V

    # Solve the transfer equation
    # dStokes/dtau_line = K_tau * Stokes - epsilon_tau

    def direct_stokes_step(current_stokes, K, epsilon, dtau):
        return current_stokes + (K @ current_stokes - epsilon[:, :, None]) * dtau

    n_steps = 20
    dtau = -line_delta_tau / n_steps
    for i in range(n_steps):
        stokes = direct_stokes_step(current_stokes=stokes, K=K_tau, epsilon=epsilon_tau, dtau=dtau)

        if i % 2 == 1:
            plotter.add_stokes(
                lambda_A=context.lambda_A,
                reference_lambda_A=context.reference_lambda_A,
                stokes=Stokes(
                    nu=context.nu,
                    I=real(stokes[:, 0, 0]),
                    Q=real(stokes[:, 1, 0]),
                    U=real(stokes[:, 2, 0]),
                    V=real(stokes[:, 3, 0]),
                ),
                stokes_reference=initial_stokes,
                label=f"FD (step #{i+1}/{n_steps})",
                linewidth=0.5,
            )

    plotter.show()


if __name__ == "__main__":
    main()
