import logging
from pathlib import Path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from numpy import real
from scipy.linalg import expm
from yatools import logging_config

from src.engine.functions.decorators import log_method
from src.common.functions import lambda_cm_to_frequency_hz, frequency_hz_to_lambda_A
from src.gui.plots.plot_stokes_profiles import StokesPlotter
from src.multi_term_atom.atomic_data.HeI import get_He_I_D3_data, fill_precomputed_He_I_D3_data
from src.multi_term_atom.object.angles import Angles
from src.multi_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.radiative_transfer_equations import MultiTermAtomRTE
from src.multi_term_atom.statistical_equilibrium_equations import MultiTermAtomSEE
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry
from src.multi_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


class Stokes:
    def __init__(self, nu: np.ndarray,  I: np.ndarray, Q: np.ndarray, U: np.ndarray, V: np.ndarray):
        self.nu = nu
        self.I = I
        self.Q = Q
        self.U = U
        self.V = V


class MultiTermAtomContext:
    def __init__(self,
                 level_registry: LevelRegistry, transition_registry: TransitionRegistry,
                 statistical_equilibrium_equations: MultiTermAtomSEE,
                    lambda_A: np.ndarray,
                    reference_lambda_A: float,):
        self.level_registry = level_registry
        self.transition_registry = transition_registry
        self.statistical_equilibrium_equations = statistical_equilibrium_equations
        self.lambda_A = lambda_A
        self.reference_lambda_A = reference_lambda_A

class ConstantPropertySlab:
    def __init__(
        self,
        multi_term_atom_context: MultiTermAtomContext,
        radiation_tensor: RadiationTensor,
        tau: float,
        chi: float,
        theta: float,
        gamma: float,
        magnetic_field_gauss: float,
        chi_B: float,
        theta_B: float,
        delta_v_thermal_cm_sm1:float,
        macroscopic_velocity_cm_sm1:float=0,
        voigt_a:float = 0,
        dtau: float = -0.1,
        initial_stokes: Union[Stokes, float, None] = None,
    ):
        self.see = multi_term_atom_context.statistical_equilibrium_equations

        self.lambda_A = multi_term_atom_context.lambda_A
        self.reference_lambda_A = multi_term_atom_context.reference_lambda_A
        self.nu = lambda_cm_to_frequency_hz(self.lambda_A * 1e-8)

        self.atmosphere_parameters = AtmosphereParameters(
            magnetic_field_gauss=magnetic_field_gauss,
            delta_v_thermal_cm_sm1=delta_v_thermal_cm_sm1,
            macroscopic_velocity_cm_sm1=macroscopic_velocity_cm_sm1,
            voigt_a=voigt_a,
        )
        self.radiation_tensor = radiation_tensor
        self.angles = Angles(
            chi=chi,
            theta=theta,
            gamma=gamma,
            chi_B=chi_B,
            theta_B=theta_B,
        )
        self.tau = tau
        # self.dtau = dtau
        self.rte = MultiTermAtomRTE(
            level_registry=multi_term_atom_context.level_registry,
            transition_registry=multi_term_atom_context.transition_registry,
            nu=self.nu,
            angles=self.angles,
            magnetic_field_gauss=self.atmosphere_parameters.magnetic_field_gauss,
        )

        self.see.add_all_equations(
            atmosphere_parameters=self.atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(
                chi_B=self.angles.chi_B, theta_B=self.angles.theta_B
            ),
        )

        if isinstance(initial_stokes, Stokes):
            assert len(initial_stokes.nu) == len(self.nu), "Initial Stokes vector must have the same frequency grid as the slab"
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = initial_stokes.I
            self.initial_stokes[:, 1, 0] = initial_stokes.Q
            self.initial_stokes[:, 2, 0] = initial_stokes.U
            self.initial_stokes[:, 3, 0] = initial_stokes.V
        elif isinstance(initial_stokes, (int, float)):
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = initial_stokes
        else:
            self.initial_stokes = np.zeros((len(self.nu), 4, 1), dtype=np.float64)
            self.initial_stokes[:, 0, 0] = 1.0

    @log_method
    def forward(self) -> Stokes:
        stokes = self.initial_stokes

        # LOS SEE
        rho = self.see.get_solution_direct()

        # LOS RT
        rtc = self.rte.compute_all_coefficients(
            atmosphere_parameters=self.atmosphere_parameters,
            rho=rho,
        )

        # DELO: S=K^-1 * epsilon, expM=expm(K*dtau), new_stokes = S + expM * (stokes - S)
        K_tau = rtc.K_tau()  # [Nν, 4, 4]
        epsilon_tau = rtc.epsilon_tau()[:, :, 0]  # [Nν, 4]
        # Stable solve for S at all ν (pinv fallback instead of det check)
        S = np.stack([
            (np.linalg.solve(K, eps)
             if np.linalg.cond(K) < 1e12
             else (np.linalg.pinv(K) @ eps))
            for K, eps in zip(K_tau, epsilon_tau)
        ])

        expM = np.stack([expm(-K * self.tau) for K in K_tau])  # [Nν,4,4]

        stokes = S[:, :, np.newaxis] + np.einsum('nij,njk->nik', expM, stokes - S[:, :, np.newaxis])

        return Stokes(
            nu=self.nu,
            I=real(stokes[:, 0, 0]),
            Q=real(stokes[:, 1, 0]),
            U=real(stokes[:, 2, 0]),
            V=real(stokes[:, 3, 0]),
        )


def create_d3_context()->MultiTermAtomContext:
    # Get atomic data
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1 = get_He_I_D3_data()
    lambda_A = np.arange(reference_lambda_A - 1, reference_lambda_A + 1, 1e-4)
    # Set up context
    see = MultiTermAtomSEE(
        level_registry=level_registry,
        transition_registry=transition_registry,
        precompute=False,
    )
    fill_precomputed_He_I_D3_data(see, root=Path(__file__).resolve().parent.parent.parent.parent.as_posix())

    context = MultiTermAtomContext(
        level_registry=level_registry,
        transition_registry=transition_registry,
        statistical_equilibrium_equations=see,
        lambda_A=lambda_A,
        reference_lambda_A=reference_lambda_A,
    )
    return context

D3_context = create_d3_context()

def radiation_tensor_NLTE_n_w_parametrized(multi_term_atom_context:MultiTermAtomContext, h_arcsec: float) -> RadiationTensor:
    return RadiationTensor(transition_registry=multi_term_atom_context.transition_registry).fill_NLTE_n_w_parametrized(h_arcsec=h_arcsec)


def plot_stokes_IQUV(stokes: Stokes, label:str, reference_lambda_A: float, show: bool = True, axs=None, normalize=True):
    if axs is None:
        fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True)
    norm = 1.0
    if normalize:
        norm = max(stokes.I)
    lambda_A = frequency_hz_to_lambda_A(stokes.nu)
    axs[0].plot(lambda_A - reference_lambda_A, stokes.I/norm, label=f"I ({label})")
    axs[0].set_ylabel("I")
    axs[0].grid(True)
    axs[1].plot(lambda_A - reference_lambda_A, stokes.Q/norm, label=f"Q ({label})")
    axs[1].set_ylabel("Q")
    axs[1].grid(True)
    axs[2].plot(lambda_A - reference_lambda_A, stokes.U/norm, label=f"U ({label})")
    axs[2].set_ylabel("U")
    axs[2].grid(True)
    axs[3].plot(lambda_A - reference_lambda_A, stokes.V/norm, label=f"V ({label})")
    axs[3].set_ylabel("V")
    axs[3].set_xlabel(r"$\Delta\lambda$ ($\AA$)")
    axs[3].grid(True)

    for ax in axs:
        ax.legend(loc="best", fontsize="x-small")

    if show:
        plt.show()
    return axs

if __name__ == '__main__':
    logging_config.init(logging.INFO)

    axs = None
    for tau in [10]:
        for b in [20000]:
            slab = ConstantPropertySlab(
                multi_term_atom_context=D3_context,
                radiation_tensor=radiation_tensor_NLTE_n_w_parametrized(D3_context, h_arcsec=10),
                tau=tau,
                chi=0,
                theta=0,
                gamma=0,
                magnetic_field_gauss=b,
                chi_B=0,
                theta_B=0,
                delta_v_thermal_cm_sm1=50_00,
                dtau=-0.1,
                initial_stokes=0,
            )
            stokes = slab.forward()

            axs = plot_stokes_IQUV(stokes, label=f"B={b} G, tau={tau}", reference_lambda_A=slab.reference_lambda_A, axs=axs, show=False, normalize=False)

    plt.show()
