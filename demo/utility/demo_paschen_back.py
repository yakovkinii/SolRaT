import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from core.steps.paschen_back import calculate_paschen_back
import unittest

from core.terms_levels_transitions.term_registry import TermRegistry


class TestDemoPaschenBack(unittest.TestCase):
    def test_demo_paschen_back(self):
        logging_config.init(logging.INFO)

        term_registry = TermRegistry()
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0.5,
            j=0.5,
            energy_cmm1=82258.9191133,
        )
        term_registry.register_term(
            beta="2p",
            l=1,
            s=0.5,
            j=1.5,
            energy_cmm1=82259.2850014,
        )
        term_registry.validate()

        level_2p = term_registry.get_level(beta="2p", l=1, s=0.5)

        energies = []
        magnetic_fields = [_ for _ in range(0, 20001, 10)]
        for magnetic_field in magnetic_fields:  # Gauss
            eigenvalues, eigenvectors = calculate_paschen_back(
                level=level_2p, magnetic_field_gauss=magnetic_field
            )
            energies.append(sorted(eigenvalues.data.values()))

        plt.plot(magnetic_fields, np.array(energies), "k")
        plt.xlabel("Magnetic field (G)")
        plt.ylabel("Energy (cm$^{-1}$)")
        plt.title("Hydrogen 2p term splitting due to Zeeman and Paschen-Back effects")
        plt.show()


if __name__ == "__main__":
    unittest.main()
