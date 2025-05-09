import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from src.two_term_atom.physics.paschen_back import calculate_paschen_back
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry


def main():
    """
    This demo shows the calculation of the Zeeman splitting for the 2p term of hydrogen,
    spanning the linear Zeeman effect regime, the intermediate fields regime,
    and the complete Paschen-Back regime.
    """

    logging_config.init(logging.INFO)

    term_registry = TermRegistry()
    term_registry.register_term(
        beta="2p",
        L=0,
        S=0,
        J=0,
        energy_cmm1=82258.9191133,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0,
        J=1,
        energy_cmm1=82259.2850014,
    )
    term_registry.validate()

    level_2p = term_registry.get_level(beta="2p", L=1, S=0)

    energies = []
    magnetic_fields = [_ for _ in range(0, 20001, 10)]
    for magnetic_field in magnetic_fields:  # Gauss
        eigenvalues, eigenvectors = calculate_paschen_back(level=level_2p, magnetic_field_gauss=magnetic_field)
        energies.append(sorted(eigenvalues.data.values()))

    plt.plot(magnetic_fields, np.array(energies), "k")
    plt.xlabel("Magnetic field (G)")
    plt.ylabel("Energy (cm$^{-1}$)")
    plt.title("Hydrogen 2p term splitting due to Zeeman and Paschen-Back effects")
    plt.show()


if __name__ == "__main__":
    main()
