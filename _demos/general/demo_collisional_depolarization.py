import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.log_setup import setup_logging

TEMPERATURE_K = 6000.0
NUMBER_DENSITY_CM3 = 1.0e11
SLAB_THICKNESS_CM = 1000e5


def log_depth_grid(z_max_cm: float, n_depth: int, min_fraction: float = 1e-6) -> np.ndarray:
    r"""
    Height grid on ``[0, z_max_cm]`` with the depth below the surface logarithmically spaced, so the
    surface optical-depth decades are resolved. ``z[0]`` is the lower boundary, ``z[-1]`` the surface.

    :param z_max_cm: slab thickness [cm].
    :param n_depth: number of depth points.
    :param min_fraction: thinnest top cell as a fraction of the slab.
    :return: sorted height grid [cm].
    """
    depth_below_surface = np.logspace(np.log10(z_max_cm * min_fraction), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth_below_surface)


def frequency_grid(transition, delta_v_thermal_cm_sm1: float) -> np.ndarray:
    r"""
    Frequency grid at ~2 points per Doppler width over +-4 Doppler widths.

    :param transition: the radiative transition.
    :param delta_v_thermal_cm_sm1: thermal+turbulent Doppler velocity [cm/s].
    :return: frequency grid [1/s].
    """
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
    step = 0.5 * delta_nu_D
    return np.arange(nu0 - 4.0 * delta_nu_D, nu0 + 4.0 * delta_nu_D + 0.5 * step, step)


def surface_alignment(depolarizing_rate_over_a_ul: float) -> float:
    r"""
    Surface upper-level fractional alignment :math:`\rho^2_0/\rho^0_0` of the collisionless-scattering
    two-level atom, with an elastic depolarizing rate :math:`D^{(2)} = (\text{ratio})\,A_{ul}` acting
    on the upper level (LL04 Sec. 7.13 / Sec. 10.6). No inelastic (de-excitation) collisions are set,
    so the atom polarizes by pure resonance scattering and :math:`D^{(2)}` is the only depolarizer.

    :param depolarizing_rate_over_a_ul: ratio :math:`D^{(2)}/A_{ul}` [dimensionless].
    :return: surface :math:`\rho^2_0/\rho^0_0`.
    """
    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    upper_level_id = transition.level_upper.level_id
    collisions.set_depolarizing_rate(
        upper_level_id, K=2, rate_sm1=depolarizing_rate_over_a_ul * transition.einstein_a_ul
    )

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=TEMPERATURE_K
    )
    nu = frequency_grid(transition, params.delta_v_thermal_cm_sm1)
    stratification = StratifiedAtmosphere(
        model=model,
        height_cm=log_depth_grid(SLAB_THICKNESS_CM, n_depth=80),
        temperature_K=TEMPERATURE_K,
        number_density_cm3=NUMBER_DENSITY_CM3,
        magnetic_field_gauss=0.0,
        velocity_cm_sm1=0.0,
        delta_v_turbulent_cm_sm1=0.0,
        voigt_a=0.0,
        continuum_to_line_ratio=0.0,
    )
    atmosphere = NLTEStratifiedAtmosphere(
        model=model,
        stratification=stratification,
        los_theta=np.pi / 2,  # tangential surface (mu = 0)
        los_chi=0.0,
        los_gamma=0.0,
        n_mu_quadrature=10,
        n_phi_quadrature=3,
        max_iterations=1000,
        tolerance=1e-8,
        ng_acceleration=True,
        ng_damping=0.7,
    )
    atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
    rho_surface = atmosphere.rho_grid[-1]
    return float(
        np.real(rho_surface(K=2, Q=0, level_id=upper_level_id))
        / np.real(rho_surface(K=0, Q=0, level_id=upper_level_id))
    )


def main():
    r"""
    Collisional depolarization of resonance-scattering polarization (LL04 Sec. 10.6).

    An elastic collision reshuffles the magnetic sublevels of the upper term without destroying the
    atom, damping the upper-level alignment :math:`\rho^2_0` (and thus the scattering polarization)
    while leaving the population :math:`\rho^0_0` untouched. For a two-level atom the alignment is
    reduced from its collisionless value by the depolarization factor
    :math:`1/(1 + D^{(2)}/A_{ul})`. This demo scans :math:`D^{(2)}/A_{ul}` for a J=0 -> J=1 scattering
    slab and overlays the SolRaT surface alignment ratio on that analytic factor, bridging the pure
    resonance-scattering limit (D^(2) = 0) toward the collisionally unpolarized (LTE) limit.

    :return: the matplotlib Figure with the normalized surface alignment versus D^(2)/A_ul and the
        analytic depolarization factor (not shown; the caller decides whether to display or save it).
    """
    setup_logging()

    depolarizing_ratios = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    alignments = np.array([surface_alignment(ratio) for ratio in depolarizing_ratios])
    normalized_alignment = alignments / alignments[0]
    analytic_factor = 1.0 / (1.0 + depolarizing_ratios)

    max_deviation = float(np.max(np.abs(normalized_alignment - analytic_factor)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(depolarizing_ratios, analytic_factor, lw=2.8, ls=(0, (1, 1)), color="k", label=r"$1/(1 + D^{(2)}/A_{ul})$")
    ax.plot(depolarizing_ratios, normalized_alignment, lw=1.2, marker="o", color="#1f77b4", label="SolRaT")
    ax.set_xscale("log")
    ax.set_xlabel(r"$D^{(2)}/A_{ul}$")
    ax.set_ylabel(r"normalized surface alignment  $\rho^2_0(D^{(2)})/\rho^2_0(0)$")
    ax.set_title("Collisional depolarization of resonance scattering (LL04 Sec. 10.6)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    print(f"Collisional depolarization: max|SolRaT - 1/(1+D2/A_ul)| = {max_deviation:.2e}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
