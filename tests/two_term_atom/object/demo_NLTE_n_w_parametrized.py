import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from src.two_term_atom.object.radiation_tensor import RadiationTensor


def main():
    """
    This recreates Fig. 4 in A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
    Here, we use the parametrized version of the n(lambda) and w(lambda) functions
    """
    logging_config.init(logging.INFO)

    radiation_tensor = RadiationTensor(
        transition_registry=...,
    )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    lambda_A = np.arange(4000, 12000, 1)
    for h_arcsec in [0, 3, 10, 40]:
        n = [radiation_tensor.n_fit(lam) for lam in lambda_A]
        w = [radiation_tensor.w_fit(lam, h_arcsec) for lam in lambda_A]
        ax[0].plot(lambda_A, n, label=rf"$n$({h_arcsec})")
        ax[1].plot(lambda_A, w, label=rf"$w$({h_arcsec})")

    ax[0].set_xlabel(r"$\lambda$ (Å)")
    ax[0].set_ylabel(r"$n$")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-7, 1e-1)

    ax[1].set_xlabel(r"$\lambda$ (Å)")
    ax[1].set_ylabel(r"$w$")
    ax[1].set_ylim(0, 0.4)

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
