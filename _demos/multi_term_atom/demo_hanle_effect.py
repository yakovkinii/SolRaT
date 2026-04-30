import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging


def main():
    r"""
    Hanle effect: depolarization of the upper-state Q=2 coherence as a function
    of magnetic field strength. Critical magnetic field is approximately
    A_ul / (Q × 2π × g_J × ν_L/G) = 4.3 G
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

    fig, ax = plt.subplots(figsize=(7, 4), num="Hanle Effect")
    ax.plot(B_values, alignments_norm, lw=2, label=r"$|\rho^2_2(J=1.5,\,J'=1.5)|\,/\,|\rho^2_2(B{=}0)|$")
    ax.axhline(1 / np.sqrt(2), color="gray", linestyle="--", lw=1, label="1/sqrt(2) level")
    ax.set_xlabel("Magnetic field B (G)")
    ax.set_ylabel("Normalised alignment modulus")
    ax.set_title("Hanle effect - upper-state Q=2 coherence vs magnetic field")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
