import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.functions import nu_larmor
from solrat.atom_model.shared.utility.log_setup import setup_logging


def main():
    r"""
    Hanle effect: depolarization of the upper-state Q=2 coherence as a function
    of magnetic field strength. Critical magnetic field is approximately
    A_ul / (Q × 2π × g_J × ν_L/G) = 4.3 G

    :return: the matplotlib Figure with the Hanle saturation curve (not shown; the caller decides
        whether to display it interactively or save it).
    """
    setup_logging()

    model = PreconfiguredModels.multi_term_atom_mock()

    angles = Angles(
        chi=0,
        theta=np.pi / 4,
        gamma=0,
        chi_B=0,
        theta_B=np.pi / 2,
    )

    radiation_tensor_mag = (
        model.RadiationTensor.from_model_config(model.config)
        .fill_NLTE_n_w_parametrized(h_arcsec=30)
        .rotate_to_magnetic_frame(angles=angles)
    )

    see = model.StatisticalEquilibriumEquations.from_model_config(model.config)

    upper_term = model.config.level_registry.get_term(beta="2p", L=1, S=0.5)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    J_align = 1.5

    B_values = np.linspace(0, 20, 40)
    alignments = []

    for B in B_values:
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=float(B),
            temperature_K=7000,
        )
        see.fill_all_equations(
            atmosphere_parameters=atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=radiation_tensor_mag,
        )
        rho = see.get_solution()
        alignments.append(
            np.abs(
                rho(
                    term_id=upper_term.term_id,
                    K=2,
                    Q=2,
                    J=J_align,
                    Jʹ=J_align,
                )
            )
        )

    alignments = np.array(alignments)
    scale = alignments[0]
    alignments_norm = alignments / scale

    # Analytic Hanle factor (LL04 eq. 10.30): |rho^2_Q(B)/rho^2_Q(0)| = 1/sqrt(1 + (Q H_u)^2), with the
    # Hanle parameter H_u = 2 pi nu_L(B) g_Ju / A_ul (Larmor precession over the upper-level lifetime),
    # here for the plotted Q = 2 coherence. g_Ju is the LS Lande factor of the 2p upper term (= 4/3).
    g_upper = 1.0 + (J_align * (J_align + 1) + 0.5 * (0.5 + 1) - 1 * (1 + 1)) / (2 * J_align * (J_align + 1))
    hanle_H = 2.0 * np.pi * nu_larmor(B_values) * g_upper / transition.einstein_a_ul
    analytic_hanle = 1.0 / np.sqrt(1.0 + (2.0 * hanle_H) ** 2)

    fig, ax = plt.subplots(figsize=(6, 6), num="Hanle Effect")
    ax.plot(B_values, alignments_norm, lw=2, label=r"SolRaT $|\rho^2_2|\,/\,|\rho^2_2(B{=}0)|$")
    ax.plot(B_values, analytic_hanle, lw=2.6, ls=(0, (1, 1)), color="k",
            label=r"LL04 eq. (10.30): $1/\sqrt{1+(Q H_u)^2}$")  # fmt: skip
    ax.axhline(1 / np.sqrt(2), color="gray", linestyle="--", lw=1, label=r"$1/\sqrt{2}$ level")
    ax.set_xlabel(r"$B$ (G)")
    ax.set_ylabel(r"$|\rho^2_2|\,/\,|\rho^2_2|_{B=0}$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    print(
        f"Hanle depolarization vs LL04 eq. (10.30): max|SolRaT - analytic| = "
        f"{float(np.max(np.abs(alignments_norm - analytic_hanle))):.2e}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
