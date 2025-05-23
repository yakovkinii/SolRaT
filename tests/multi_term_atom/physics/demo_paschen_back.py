import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from src.multi_term_atom.physics.paschen_back import calculate_paschen_back
from src.multi_term_atom.terms_levels_transitions.level_registry import LevelRegistry


def main():
    """
    This demo shows the calculation of the Zeeman splitting for the 2p term of hydrogen,
    spanning the linear Zeeman effect regime, the intermediate fields regime,
    and the complete Paschen-Back regime.
    """

    logging_config.init(logging.INFO)

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

    plt.plot(magnetic_fields, np.array(energies), "k")
    plt.xlabel("Magnetic field (G)")
    plt.ylabel("Energy (cm$^{-1}$)")
    plt.title("Hydrogen 2p term splitting due to Zeeman and Paschen-Back effects")
    plt.show()


if __name__ == "__main__":
    main()
