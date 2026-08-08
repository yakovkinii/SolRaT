import matplotlib.pyplot as plt
import numpy as np

from solrat.atom_model.multi_term_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_term_atom_model.utility.paschen_back import calculate_paschen_back
from solrat.atom_model.shared.utility.log_setup import setup_logging


def main():
    """
    This demo shows the calculation of the Zeeman splitting for the 2p term of hydrogen,
    spanning the linear Zeeman effect regime, the intermediate fields regime,
    and the complete Paschen-Back regime.

    :return: the matplotlib Figure with the term-splitting diagram (not shown; the caller decides
        whether to display it interactively or save it).
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
    magnetic_fields = [_ for _ in range(0, 20001, 10)]
    for magnetic_field in magnetic_fields:  # Gauss
        eigenvalues, eigenvectors = calculate_paschen_back(term=term_2p, magnetic_field_gauss=magnetic_field)
        energies.append(sorted(eigenvalues.data.values()))

    fig, ax = plt.subplots()
    ax.plot(magnetic_fields, np.array(energies), "k")
    ax.set_xlabel("Magnetic field (G)")
    ax.set_ylabel("Energy (cm$^{-1}$)")
    ax.set_title("Hydrogen 2p term splitting due to Zeeman and Paschen-Back effects")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
