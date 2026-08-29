import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.multi_term_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    PrescribedRadiationStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
    nu_larmor,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.voigt_profile import voigt


def zeeman_triplet_profiles(v: np.ndarray, a: float, v_B: float):
    r"""
    Absorption (phi) and dispersion (psi) profiles of the pi and the two sigma components of a
    normal Zeeman triplet (LL04 eqs. 5.36-5.37). Returns ``((phi_p, phi_b, phi_r),
    (psi_p, psi_b, psi_r))``.

    The blue sigma sits at nu0 + nu_L (peak at v = +v_B), the red sigma at nu0 - nu_L (peak at
    v = -v_B). SolRaT's complex Voigt gives H = Re(voigt) (even) and the Faraday-Voigt L = Im(voigt)
    (odd); the absorption is phi = H(v - v_center), and LL04's dispersion is psi ~ (nu_center - nu),
    i.e. psi = -L(v - v_center).

    :param v: reduced frequency (nu - nu0) / delta_nu_D [dimensionless]; positive v = higher frequency.
    :param a: Voigt damping parameter [dimensionless].
    :param v_B: Zeeman splitting in Doppler widths [dimensionless].
    :return: tuple ((phi_p, phi_b, phi_r), (psi_p, psi_b, psi_r)) of arrays over ``v``.
    """
    w_p = voigt(nu=v, a=a)  # pi: center nu0
    w_b = voigt(nu=v - v_B, a=a)  # blue sigma: center nu0 + nu_L (higher frequency)
    w_r = voigt(nu=v + v_B, a=a)  # red sigma: center nu0 - nu_L (lower frequency)
    phi = (np.real(w_p), np.real(w_b), np.real(w_r))
    psi = (-np.imag(w_p), -np.imag(w_b), -np.imag(w_r))
    return phi, psi


def zeeman_triplet_line_coefficients(v: np.ndarray, a: float, v_B: float, theta_B: float, chi_B: float):
    r"""
    The seven line propagation-matrix coefficients of a normal Zeeman triplet, LL04 eq. (5.36) -- the
    coefficients of the line-to-continuum ratio eta_0 -- returned as
    ``(eta_I_line, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V)``. At zero field and line center
    ``eta_I_line = 1``. The Stokes-V sign is LL04's: eta_V = (1/2)(phi_red - phi_blue) cos(theta_B).

    :param v: reduced frequency (nu - nu0) / delta_nu_D [dimensionless].
    :param a: Voigt damping parameter [dimensionless].
    :param v_B: Zeeman splitting in Doppler widths [dimensionless].
    :param theta_B: angle between the magnetic field and the line of sight [rad].
    :param chi_B: magnetic-field azimuth [rad].
    :return: tuple of seven arrays over ``v``.
    """
    (phi_p, phi_b, phi_r), (psi_p, psi_b, psi_r) = zeeman_triplet_profiles(v, a, v_B)
    sin2, cos = np.sin(theta_B) ** 2, np.cos(theta_B)
    cos2x, sin2x = np.cos(2 * chi_B), np.sin(2 * chi_B)
    phi_sigma, psi_sigma = 0.5 * (phi_b + phi_r), 0.5 * (psi_b + psi_r)

    eta_I_line = 0.5 * (phi_p * sin2 + phi_sigma * (1.0 + cos**2))
    eta_Q = 0.5 * (phi_p - phi_sigma) * sin2 * cos2x
    eta_U = 0.5 * (phi_p - phi_sigma) * sin2 * sin2x
    eta_V = 0.5 * (phi_r - phi_b) * cos
    rho_Q = 0.5 * (psi_p - psi_sigma) * sin2 * cos2x
    rho_U = 0.5 * (psi_p - psi_sigma) * sin2 * sin2x
    rho_V = 0.5 * (psi_r - psi_b) * cos
    return eta_I_line, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V


def unno_rachkovsky_emergent_stokes(
    v, a, v_B, theta_B, chi_B, eta_0, mu, source_0, source_1, magneto_optical=True, normalization="center"
):
    r"""
    Analytic Unno-Rachkovsky emergent Stokes ``(I, Q, U, V)`` for a Milne-Eddington atmosphere and a
    normal Zeeman triplet (LL04 eq. 9.109): I(0) = S0 u + mu S1 Khat^{-1} u, u = (1,0,0,0)^T, with
    Khat = 1 + eta_0 * (line coefficients).

    :param v: reduced frequency (nu - nu0) / delta_nu_D [dimensionless].
    :param a: Voigt damping parameter [dimensionless].
    :param v_B: Zeeman splitting in Doppler widths [dimensionless].
    :param theta_B: angle between the magnetic field and the line of sight [rad].
    :param chi_B: magnetic-field azimuth [rad].
    :param eta_0: line-center-to-continuum opacity ratio [dimensionless].
    :param mu: emergent direction cosine [dimensionless].
    :param source_0: surface source function S0 [source-function units].
    :param source_1: source-function gradient S1 = dS/dtau_c along the line of sight [source-function units].
    :param magneto_optical: if False, drop the dispersion terms rho_Q, rho_U, rho_V (pure Unno limit).
    :param normalization: ``"center"`` leaves eta_I_line = 1 at zero-field line center; ``"max"`` divides
        the line coefficients by the maximum of eta_I_line over the grid, matching SolRaT's ``rtc.K_tau()``.
    :return: tuple (I, Q, U, V) of arrays over ``v``.
    """
    eta_I_line, eta_Q, eta_U, eta_V, rho_Q, rho_U, rho_V = zeeman_triplet_line_coefficients(v, a, v_B, theta_B, chi_B)
    if not magneto_optical:
        rho_Q, rho_U, rho_V = np.zeros_like(rho_Q), np.zeros_like(rho_U), np.zeros_like(rho_V)

    scale = np.max(eta_I_line) if normalization == "max" else 1.0
    factor = eta_0 / scale

    n = len(v)
    K = np.zeros((n, 4, 4), dtype=np.float64)
    K[:, 0, 0] = K[:, 1, 1] = K[:, 2, 2] = K[:, 3, 3] = 1.0 + factor * eta_I_line
    K[:, 0, 1] = K[:, 1, 0] = factor * eta_Q
    K[:, 0, 2] = K[:, 2, 0] = factor * eta_U
    K[:, 0, 3] = K[:, 3, 0] = factor * eta_V
    K[:, 1, 2], K[:, 2, 1] = factor * rho_V, -factor * rho_V
    K[:, 1, 3], K[:, 3, 1] = -factor * rho_U, factor * rho_U
    K[:, 2, 3], K[:, 3, 2] = factor * rho_Q, -factor * rho_Q

    rhs = np.zeros((n, 4, 1), dtype=np.float64)
    rhs[:, 0, 0] = 1.0
    k_inv_u = np.linalg.solve(K, rhs)[:, :, 0]
    e0 = np.array([1.0, 0.0, 0.0, 0.0])
    stokes = source_0 * e0[np.newaxis, :] + mu * source_1 * k_inv_u
    return stokes[:, 0], stokes[:, 1], stokes[:, 2], stokes[:, 3]


def build_normal_triplet_lte_model():
    r"""
    Build a ^1S_0 -> ^1P_1 LTE multi-term atom -- a clean normal Zeeman triplet with Lande factor
    g_u = 1 -- and return ``(model, nu0_sm1, reference_lambda_A_air)``.

    :return: tuple (model, transition frequency [1/s], reference air wavelength [Angstrom]).
    """
    upper_energy_cmm1 = 20_000.0  # ~5000 A transition
    level_registry = LevelRegistry()
    level_registry.register_level(beta="lower", L=0, S=0, J=0, energy_cmm1=0.0)
    level_registry.register_level(beta="upper", L=1, S=0, J=1, energy_cmm1=upper_energy_cmm1)
    level_registry.validate()

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        term_upper=level_registry.get_term(beta="upper", L=1, S=0),
        term_lower=level_registry.get_term(beta="lower", L=0, S=0),
        einstein_a_ul_sm1=1e7,
    )

    nu0 = energy_cmm1_to_frequency_sm1(upper_energy_cmm1)
    reference_lambda_A_air = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu0))
    config = MultiTermAtomConfig(
        level_registry=level_registry,
        transition_registry=transition_registry,
        reference_lambda_A_air=reference_lambda_A_air,
        atomic_mass_amu=56.0,
    )
    return Models.multi_term_atom_lte().configure(config=config), nu0, reference_lambda_A_air


def line_center_opacity_per_atom(model, nu, atmosphere_parameters, angles):
    r"""
    Maximum line opacity for ``N = 1`` along the validation ray.
    """
    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)
    see.fill_all_equations(
        atmosphere_parameters=atmosphere_parameters,
        radiation_tensor_in_magnetic_frame=model.RadiationTensor(),
    )
    rte = model.RadiativeTransferEquations.from_model_config(model.config, nu=nu)
    rte.N = 1.0
    rtc = rte.calculate_all_coefficients(
        atmosphere_parameters=atmosphere_parameters,
        angles=angles,
        rho=see.get_solution(),
    )
    return float(np.max(np.abs(rtc.get_eta_I())))


def main():
    r"""
    Benchmark fully numerical stratified prescribed-JKQ transfer against the analytic
    Unno-Rachkovsky (Milne-Eddington) solution for a normal Zeeman triplet (LL04 eq. 9.109) at
    several field strengths. The Stokes-V sign follows LL04 eq. 5.36.

    :return: matplotlib Figure.
    """
    setup_logging()

    temperature_K = 6000.0
    delta_v_turbulent_cm_sm1 = 2.0e5
    voigt_a = 0.05
    theta_B = np.deg2rad(60.0)
    chi_B = np.deg2rad(0.0)
    eta_0 = 10.0
    source_0, source_1 = 1.0, 3.0
    tau_c_total = 30.0

    model, nu0, reference_lambda_A_air = build_normal_triplet_lte_model()
    nu = get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.35,
        upper_wavelength_A=reference_lambda_A_air + 0.35,
        step_A=1e-3,
    )
    angles = Angles(chi=0.0, theta=0.0, gamma=0.0, chi_B=chi_B, theta_B=theta_B)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axis_labels = ["$I$", "$Q/I$", "$U/I$", "$V/I$"]
    field_values_gauss = [500.0, 1500.0, 3000.0]
    colors = ["k", "#d62728", "#2ca02c"]
    stokes_residuals = []

    for magnetic_field_gauss, color in zip(field_values_gauss, colors):
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=magnetic_field_gauss,
            temperature_K=temperature_K,
            delta_v_turbulent_cm_sm1=delta_v_turbulent_cm_sm1,
            voigt_a=voigt_a,
        )
        line_center_opacity = line_center_opacity_per_atom(model, nu, atmosphere_parameters, angles)
        continuum_opacity = line_center_opacity / eta_0
        height = tau_c_total / continuum_opacity
        z = np.linspace(0.0, height, 220)
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=z,
            temperature_K=temperature_K,
            number_density_cm3=1.0,
            magnetic_field_gauss=magnetic_field_gauss,
            theta_B=theta_B,
            chi_B=chi_B,
            delta_v_turbulent_cm_sm1=delta_v_turbulent_cm_sm1,
            voigt_a=voigt_a,
            continuum_opacity_cm_m1=continuum_opacity,
        )
        initial_stokes = Stokes(
            nu=nu,
            I=np.full_like(nu, source_0 + source_1 * tau_c_total),
            Q=np.zeros_like(nu),
            U=np.zeros_like(nu),
            V=np.zeros_like(nu),
        )
        atmosphere = PrescribedRadiationStratifiedAtmosphere(
            model=model,
            radiation_tensor=model.RadiationTensor(),
            stratification=stratification,
            los_theta=0.0,
            los_chi=0.0,
            los_gamma=0.0,
            source_function_I=lambda _z, tau_c: source_0 + source_1 * tau_c,
            transfer_scheme="delo_linear",
        )
        solrat_stokes = atmosphere.forward(initial_stokes=initial_stokes)

        delta_nu_D = nu0 * atmosphere_parameters.delta_v_thermal_cm_sm1 / c_cm_sm1
        v = (nu - nu0) / delta_nu_D
        v_B = float(nu_larmor(np.array(magnetic_field_gauss))) / delta_nu_D  # g_u = 1 for ^1P_1
        stokes_I_a, stokes_Q_a, stokes_U_a, stokes_V_a = unno_rachkovsky_emergent_stokes(
            v=v,
            a=voigt_a,
            v_B=v_B,
            theta_B=theta_B,
            chi_B=chi_B,
            eta_0=eta_0,
            mu=1.0,
            source_0=source_0,
            source_1=source_1,
            normalization="max",
        )
        solrat_i_norm = solrat_stokes.I / np.max(solrat_stokes.I)
        analytic_i_norm = stokes_I_a / np.max(stokes_I_a)
        panels = [
            (solrat_i_norm, analytic_i_norm),
            (solrat_stokes.Q / solrat_stokes.I, stokes_Q_a / stokes_I_a),
            (solrat_stokes.U / solrat_stokes.I, stokes_U_a / stokes_I_a),
            (solrat_stokes.V / solrat_stokes.I, stokes_V_a / stokes_I_a),
        ]
        for solrat_curve, analytic_curve in panels:
            stokes_residuals.append(float(np.mean((solrat_curve - analytic_curve) ** 2)))
        for ax, (solrat_curve, analytic_curve) in zip(axes.ravel(), panels):
            ax.plot(v, solrat_curve, lw=0.9, color=color)
            ax.plot(v, analytic_curve, lw=2.4, ls=(0, (1, 1)), color=color)

    for ax, label in zip(axes.ravel(), axis_labels):
        ax.set_ylabel(label)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.grid(color="0.88", linewidth=0.5, alpha=0.7)
    for ax in axes[1]:
        ax.set_xlabel(r"$(\nu - \nu_0)/\Delta\nu_D$")
    axes[0, 0].set_ylabel(r"$I\,/\,I_{\max}$")
    style_key = [
        Line2D([], [], color="k", lw=0.9, label="SolRaT"),
        Line2D([], [], color="k", lw=2.4, ls=(0, (1, 1)), label="Unno-Rachkovsky"),
    ]
    field_key = [
        Line2D([], [], color=color, lw=2.4, label=f"B = {b:.0f} G") for b, color in zip(field_values_gauss, colors)
    ]
    axes[0, 0].legend(handles=style_key, fontsize=11, loc="best")
    axes[0, 1].legend(handles=field_key, fontsize=11, loc="best")
    fig.align_ylabels(axes.ravel())
    fig.tight_layout()
    print(
        f"Stratified prescribed-JKQ numerical RTE vs analytic Unno-Rachkovsky: "
        f"RMS Delta Stokes = {float(np.sqrt(np.mean(stokes_residuals))):.2e} "
        f"(over B = {[int(b) for b in field_values_gauss]} G)"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
