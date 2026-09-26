import matplotlib.pyplot as plt
import numpy as np

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.utility.paschen_back import _g_ls, calculate_paschen_back
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, mu0_erg_gaussm1
from solrat.atom_model.shared.utility.log_setup import setup_logging

# --- Field grid -------------------------------------------------------------
B_MAX_GAUSS = 20000
B_STEP_GAUSS = 10

# --- Regime boundaries [G]: adjust these two to move the dividing lines ------
# Below B_LINEAR_ZEEMAN_MAX_GAUSS the splitting is linear in B.
# Above B_INCOMPLETE_PB_MAX_GAUSS the term is in the complete Paschen-Back regime.
B_LINEAR_ZEEMAN_MAX_GAUSS = 3000.0
B_INCOMPLETE_PB_MAX_GAUSS = 12000.0

# --- Regime annotation style ------------------------------------------------
REGIME_COLOR = "b"  # gray-blue
REGIME_LINE_WIDTH = 0.9
REGIME_SHADE_ALPHA = 0.10  # shading of the linear-Zeeman band
REGIME_LABEL_Y = 1.02  # axes fraction; use ~0.97 with REGIME_LABEL_VA="top" to put labels inside
REGIME_LABEL_VA = "bottom"
REGIME_LABEL_SIZE = 11
REGIME_LABELS = ("Linear\nZeeman", "Incomplete\nPaschen-Back", "Complete\nPaschen-Back")


def main():
    r"""
    Zeeman splitting of the hydrogen 2p term across the linear-Zeeman, incomplete, and complete
    Paschen-Back regimes.

    The exact eigenvalues of the spin-orbit plus magnetic Hamiltonian are drawn as solid curves.
    The linear-Zeeman sublevel energies :math:`E_J + g_J \mu_0 B M / hc` are overplotted dashed, so
    the field at which the two separate is visible directly. Vertical lines split the axis into the
    three regimes and the linear-Zeeman band is shaded.

    :return: matplotlib Figure.
    """

    setup_logging()

    level_registry = LevelRegistry()
    level_registry.register_level(
        beta="2p",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=82258.9191133,
    )
    level_registry.register_level(
        beta="2p",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=82259.2850014,
    )
    level_registry.validate()

    term_2p = level_registry.get_term(beta="2p", L=1, S=0.5)

    energies = []
    magnetic_fields = [_ for _ in range(0, B_MAX_GAUSS + 1, B_STEP_GAUSS)]
    for magnetic_field in magnetic_fields:  # Gauss
        eigenvalues, eigenvectors = calculate_paschen_back(term=term_2p, magnetic_field_gauss=magnetic_field)
        energies.append(sorted(eigenvalues.data.values()))

    # Linear-Zeeman sublevel energies: the diagonal of the same Hamiltonian, with the
    # off-diagonal J-coupling dropped.
    B = np.asarray(magnetic_fields, dtype=float)
    B = B[B <= 7000]
    mu0b_cm = mu0_erg_gaussm1 * B / h_erg_s / c_cm_sm1  # mu_0 * B in cm-1
    linear_zeeman = []
    for level in term_2p.levels:
        g = _g_ls(term_2p.L, term_2p.S, level.J, artificial_S_scale=term_2p.artificial_S_scale)
        for M in np.arange(-level.J, level.J + 1):
            linear_zeeman.append(level.energy_cmm1 + mu0b_cm * g * M)

    fig, ax = plt.subplots()

    # Regime bands: shade the linear-Zeeman range and split the axis.
    ax.axvspan(0, B_LINEAR_ZEEMAN_MAX_GAUSS, color=REGIME_COLOR, alpha=REGIME_SHADE_ALPHA, lw=0, zorder=0)
    for boundary in (B_LINEAR_ZEEMAN_MAX_GAUSS, B_INCOMPLETE_PB_MAX_GAUSS):
        if 0 < boundary < B_MAX_GAUSS:
            ax.axvline(
                boundary,
                color=REGIME_COLOR,
                lw=REGIME_LINE_WIDTH,
                ls="-",
                alpha=0.8,
                zorder=1,
            )

    ax.plot(magnetic_fields, np.array(energies), "k", lw=1.0, zorder=3)
    for curve in linear_zeeman:
        ax.plot(B, curve, color="k", lw=0.9, ls="--", zorder=2)

    # Proxy handles so each style appears once in the legend.
    ax.plot([], [], "k", lw=1.0, label="Exact diagonalization")
    ax.plot([], [], color="k", lw=0.9, ls="--", label="Linear Zeeman")
    # ax.legend(loc="center left", fontsize=7, frameon=False)
    ax.legend(framealpha=1.0)

    # Regime labels, centred on each band.
    edges = [0.0, B_LINEAR_ZEEMAN_MAX_GAUSS, B_INCOMPLETE_PB_MAX_GAUSS, float(B_MAX_GAUSS)]
    for left, right, label in zip(edges[:-1], edges[1:], REGIME_LABELS):
        if right <= left:
            continue
        ax.text(
            0.5 * (left + right),
            REGIME_LABEL_Y,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va=REGIME_LABEL_VA,
            fontsize=REGIME_LABEL_SIZE,
            color=REGIME_COLOR,
        )

    ax.set_xlim(0, B_MAX_GAUSS)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_xlabel(r"$B$ (G)")
    ax.set_ylabel(r"$E$ (cm$^{-1}$)")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
