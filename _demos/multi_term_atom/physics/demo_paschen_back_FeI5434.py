import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from src.multi_term_atom.physics.paschen_back import (
    calculate_paschen_back,
    get_artificial_S_scale_from_term_g,
)
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
        beta="a5F",
        L=3,  # F term  -> L = 3
        S=2.0,  # 2S+1 = 5 -> S = 2
        J=1.0,  # a^5F_1
        energy_cmm1=8154.713,  # NIST: 3d^7(4F)4s a^5F1 level
    )
    level_registry.register_level(
        beta="a5F",
        L=3,  # F term  -> L = 3
        S=2.0,  # 2S+1 = 5 -> S = 2
        J=2.0,  # a^5F_1
        energy_cmm1=7985.785,  # NIST: 3d^7(4F)4s a^5F1 level
    )
    level_registry.register_level(
        beta="a5F",
        L=3,  # F term  -> L = 3
        S=2.0,  # 2S+1 = 5 -> S = 2
        J=3.0,  # a^5F_1
        energy_cmm1=7728.060,  # NIST: 3d^7(4F)4s a^5F1 level
    )
    level_registry.register_level(
        beta="a5F",
        L=3,  # F term  -> L = 3
        S=2.0,  # 2S+1 = 5 -> S = 2
        J=4.0,  # a^5F_1
        energy_cmm1=7376.764,  # NIST: 3d^7(4F)4s a^5F1 level
    )
    level_registry.register_level(
        beta="a5F",
        L=3,  # F term  -> L = 3
        S=2.0,  # 2S+1 = 5 -> S = 2
        J=5.0,  # a^5F_1
        energy_cmm1=6928.268,  # NIST: 3d^7(4F)4s a^5F1 level
    )
    level_registry.validate()

    term = level_registry.get_term(beta="a5F", L=3, S=2.0)

    energies = []
    magnetic_fields = [_ for _ in range(0, 20001, 1000)]
    for magnetic_field in magnetic_fields:  # Gauss
        eigenvalues, eigenvectors = calculate_paschen_back(term=term, magnetic_field_gauss=magnetic_field)
        energies.append(sorted(eigenvalues.data.values()))

    plt.plot(magnetic_fields, np.array(energies), "k", label="Pure LS")

    term.artificial_S_scale = get_artificial_S_scale_from_term_g(g=-0.014, L=3, S=2, J=1)
    energies = []
    for magnetic_field in magnetic_fields:  # Gauss
        eigenvalues, eigenvectors = calculate_paschen_back(term=term, magnetic_field_gauss=magnetic_field)
        energies.append(sorted(eigenvalues.data.values()))

    plt.plot(magnetic_fields, np.array(energies), "r", label="Adjusted LS")
    plt.ylim(8154.7, 8154.74)
    plt.xlabel("Magnetic field (G)")
    plt.ylabel("Energy (cm$^{-1}$)")
    plt.title(
        "FeI 5434: $J=1$ Lower term Zeeman splitting: \nPure LS (black) and LS with S scaled to mimic g=-0.014 (red)"
    )
    plt.show()


if __name__ == "__main__":
    main()
