import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.utility.paschen_back import calculate_paschen_back
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, mu0_erg_gaussm1
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.atom_model.shared.utility.wigner_3j_6j_9j import wigner_3j

MAGNETIC_FIELD_GAUSS = 500.0  # weak field: the pattern is in the linear-Zeeman regime
J_LOWER = 0.5  # 2S_1/2
J_UPPER = 1.5  # 2P_3/2 (the D2-like branch)


def build_terms():
    r"""
    Build the lower ^2S_{1/2} and upper ^2P_{3/2,1/2} terms of a D2-like doublet.

    :return: tuple (lower Term, upper Term).
    """
    level_registry = LevelRegistry()
    level_registry.register_level(beta="lower", L=0, S=0.5, J=0.5, energy_cmm1=0.0)
    level_registry.register_level(beta="upper", L=1, S=0.5, J=0.5, energy_cmm1=20_000.0)
    level_registry.register_level(beta="upper", L=1, S=0.5, J=1.5, energy_cmm1=20_020.0)
    level_registry.validate()
    return level_registry.get_term(beta="lower", L=0, S=0.5), level_registry.get_term(beta="upper", L=1, S=0.5)


def zeeman_components(term_lower, term_upper, magnetic_field_gauss: float):
    r"""
    Zeeman components of the ``term_lower`` (J_l) -> ``term_upper`` (J_u) branch: for every allowed
    M_l -> M_u transition (Delta M = q in {-1, 0, +1}), the displacement from line center in units of
    the Lorentz splitting and the relative strength. The displacements come from SolRaT's
    Paschen-Back eigenvalues (LL04 eq. 3.61); the strengths are the LS dipole strengths
    3 (J_u 1 J_l; -M_u q M_l)^2 (LL04 Table 3.1 / eq. 3.60).

    :param term_lower: lower Term.
    :param term_upper: upper Term.
    :param magnetic_field_gauss: field strength [G].
    :return: tuple (q_array, displacement_over_lorentz_array, strength_array).
    """
    eigenvalues_lower, _ = calculate_paschen_back(term=term_lower, magnetic_field_gauss=magnetic_field_gauss)
    eigenvalues_upper, _ = calculate_paschen_back(term=term_upper, magnetic_field_gauss=magnetic_field_gauss)
    lorentz_unit_cmm1 = mu0_erg_gaussm1 * magnetic_field_gauss / h_erg_s / c_cm_sm1
    line_center_cmm1 = term_upper.get_level(J_UPPER).energy_cmm1 - term_lower.get_level(J_LOWER).energy_cmm1

    q_values, displacements, strengths = [], [], []
    for m_lower in np.arange(-J_LOWER, J_LOWER + 1):
        for m_upper in np.arange(-J_UPPER, J_UPPER + 1):
            q = m_upper - m_lower
            if abs(q) > 1:
                continue
            displacement_cmm1 = (
                eigenvalues_upper(j=J_UPPER, M=m_upper) - eigenvalues_lower(j=J_LOWER, M=m_lower) - line_center_cmm1
            )
            strength = 3.0 * wigner_3j(J_UPPER, 1, J_LOWER, -m_upper, q, m_lower) ** 2
            q_values.append(q)
            displacements.append(displacement_cmm1 / lorentz_unit_cmm1)
            strengths.append(strength)
    return np.array(q_values), np.array(displacements), np.array(strengths)


def main():
    r"""
    Zeeman pattern of an anomalous multiplet (:math:`^2S_{1/2} \to {}^2P_{3/2}`, D2-like): the
    :math:`\pi` and :math:`\sigma^\pm` component positions and relative strengths, against the
    linear-Zeeman positions (LL04 Sec. 3.3).

    :return: matplotlib Figure.
    """
    setup_logging()

    term_lower, term_upper = build_terms()
    q_values, displacements, strengths = zeeman_components(term_lower, term_upper, MAGNETIC_FIELD_GAUSS)

    # Linear-Zeeman reference positions (LS Lande factor, LL04 eq. 3.8), independent of the code under test.
    lande = lambda L, S, J: 1.0 + 0.5 * (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (J * (J + 1))  # noqa: E731
    g_lower = lande(term_lower.L, term_lower.S, J_LOWER)
    g_upper = lande(term_upper.L, term_upper.S, J_UPPER)
    analytic_positions = []
    for m_lower in np.arange(-J_LOWER, J_LOWER + 1):
        for m_upper in np.arange(-J_UPPER, J_UPPER + 1):
            if abs(m_upper - m_lower) > 1:
                continue
            analytic_positions.append(g_upper * m_upper - g_lower * m_lower)
    analytic_positions = np.array(analytic_positions)

    position_rms = float(np.sqrt(np.mean((displacements - analytic_positions) ** 2)))

    colors = {(-1): "#d62728", 0: "#1f77b4", 1: "#2ca02c"}
    labels = {(-1): r"$\sigma_-$ ($\Delta M = -1$)", 0: r"$\pi$ ($\Delta M = 0$)", 1: r"$\sigma_+$ ($\Delta M = +1$)"}
    fig, ax = plt.subplots(figsize=(8, 4))
    seen = set()
    for q, position, strength in zip(q_values, displacements, strengths):
        label = labels[q] if q not in seen else None
        seen.add(q)
        ax.vlines(position, 0.0, strength, color=colors[q], lw=2, label=label)
    ax.plot(analytic_positions, np.zeros_like(analytic_positions), "kx", label="linear Zeeman (positions)")
    ax.axvline(0.0, color="0.7", lw=0.6)
    ax.set_xlabel(r"displacement from line center  $(\nu - \nu_0)\,/\,\nu_L$  (Lorentz units)")
    ax.set_ylabel("relative strength")
    ax.set_title(r"Zeeman pattern of $^2S_{1/2}\to{}^2P_{3/2}$ (anomalous, D2-like)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    print(
        f"Zeeman pattern (B = {MAGNETIC_FIELD_GAUSS:.0f} G): RMS SolRaT - linear Zeeman positions = "
        f"{position_rms:.2e} Lorentz units"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
