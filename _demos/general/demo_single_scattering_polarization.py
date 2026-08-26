import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging

UPPER_ENERGY_CMM1 = 20_000.0  # ~5000 A, inside the n(lambda)/w(lambda) anisotropy fit range


def build_j0_j1_atom(reference_lambda_A_air: float):
    r"""
    A J = 0 -> J = 1 two-level (multi-level) atom, the textbook resonance-scattering transition.

    :param reference_lambda_A_air: air reference wavelength [Angstrom].
    :return: configured multi-level Model.
    """
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="lower", J=0, energy_cmm1=0.0, g=1.0)
    level_registry.register_level(alpha="upper", J=1, energy_cmm1=UPPER_ENERGY_CMM1, g=1.0)
    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="upper", J=1),
        level_lower=level_registry.get_level(alpha="lower", J=0),
        einstein_a_ul_sm1=1e7,
    )
    config = MultiLevelAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        atomic_mass_amu=56.0,
        reference_lambda_A_air=reference_lambda_A_air,
        collisions=None,
    )
    return Models.multi_level_atom().configure(config=config)


def main():
    r"""
    Single-scattering (resonance) polarization of a :math:`J=0 \to 1` line versus scattering angle,
    against the Rayleigh :math:`\sin^2\theta` law (LL04 Sec. 10.2 / 5.8). Angular-shape check
    (normalized), not an absolute-amplitude validation.

    :return: matplotlib Figure.
    """
    setup_logging()

    nu0 = energy_cmm1_to_frequency_sm1(UPPER_ENERGY_CMM1)
    reference_lambda_A_air = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu0))
    model = build_j0_j1_atom(reference_lambda_A_air)
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.2,
        upper_wavelength_A=reference_lambda_A_air + 0.2,
        step_A=2e-3,
    )
    line_center_index = int(np.argmin(np.abs(nu - nu0)))

    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=6000.0, delta_v_turbulent_cm_sm1=2.0e5
    )
    field_frame_angles = Angles(chi=0.0, theta=0.0, gamma=0.0, chi_B=0.0, theta_B=0.0)
    radiation_tensor = (
        model.RadiationTensor.from_model_config(model.config)
        .fill_NLTE_n_w_allen(h_arcsec=30)
        .rotate_to_magnetic_frame(angles=field_frame_angles)
    )
    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters, radiation_tensor_in_magnetic_frame=radiation_tensor
    )
    rho = see.get_solution()

    rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
    scattering_angles = np.linspace(0.0, np.pi, 37)
    qi_line_center = []
    for theta in scattering_angles:
        rtc = rte.calculate_all_coefficients(
            atmosphere_parameters=atmosphere_parameters,
            rho=rho,
            angles=Angles(chi=0.0, theta=float(theta), gamma=0.0, chi_B=0.0, theta_B=0.0),
        )
        qi_line_center.append(rtc.get_eta_Q()[line_center_index] / rtc.get_eta_I()[line_center_index])
    qi_line_center = np.array(qi_line_center)

    rayleigh = np.sin(scattering_angles) ** 2
    normalization = np.max(np.abs(qi_line_center))
    shape_residual = np.abs(qi_line_center) / normalization - rayleigh / np.max(rayleigh)
    shape_rms = float(np.sqrt(np.mean(shape_residual**2)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        np.rad2deg(scattering_angles),
        np.abs(qi_line_center) / normalization,
        lw=1.2,
        marker="o",
        color="#1f77b4",
        label=r"SolRaT $|\eta_Q/\eta_I|$ (normalized)",
    )
    ax.plot(
        np.rad2deg(scattering_angles),
        rayleigh / np.max(rayleigh),
        lw=2.8,
        ls=(0, (1, 1)),
        color="k",
        label=r"Rayleigh $\sin^2\theta$ (normalized)",
    )
    ax.set_xlabel(r"scattering angle $\theta$ (deg)")
    ax.set_ylabel("normalized line-center polarization")
    ax.set_title(r"Single-scattering polarization of a $J=0\to1$ line (LL04 Sec. 10.2)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    print(
        f"Single-scattering polarization: RMS SolRaT - Rayleigh sin^2 (normalized shape) = "
        f"{shape_rms:.2e}; peak |Q/I| = {normalization:.3e}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
