import matplotlib.pyplot as plt
import numpy as np

from core.utility.paschen_back import paschen_back_diagonalization
import unittest


def energies_hydrogen_2p():
    """
    Dictionary with energies for the 2p level of H I with different J.
    Units: cm-1
    """
    return {
        0.5: 82258.9191133,
        1.5: 82259.2850014,
    }


class TestDemoPaschenBack(unittest.TestCase):
    def test_demo_paschen_back(self):
        unperturbed_energies = energies_hydrogen_2p()

        energies = []
        magnetic_fields = [_ for _ in range(0, 20001, 10)]
        for magnetic_field in magnetic_fields:  # Gauss
            eigenvalues, eigenvectors = paschen_back_diagonalization(
                l=1, s=0.5, b=magnetic_field, e=unperturbed_energies
            )
            energies.append(sorted(eigenvalues.data.values()))

        plt.plot(magnetic_fields, np.array(energies), "k")
        plt.xlabel("Magnetic field (G)")
        plt.ylabel("Energy (cm$^{-1}$)")
        plt.title("Hydrogen 2p term splitting due to Zeeman and Paschen-Back effects")
        plt.show()


if __name__ == "__main__":
    unittest.main()
