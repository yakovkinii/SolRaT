import logging

import matplotlib.pyplot as plt
import numpy as np
from yatools import logging_config

from src.multi_term_atom.atomic_data.HeI import get_He_I_D3_data
from src.multi_term_atom.object.radiation_tensor import RadiationTensor


def main():
    """
    This recreates Fig. 4 in A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
    Here, we use the parametrized version of the n(lambda) and w(lambda) functions
    """
    logging_config.init(logging.INFO)
    # Load the atomic data for He I D3
    level_registry, transition_registry, reference_lambda_A, reference_nu_sm1, atomic_mass_amu = get_He_I_D3_data()

    radiation_tensor = RadiationTensor(
        transition_registry=transition_registry,
    )

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig.suptitle("Anisotropic parametrization of $J_0^0$ and $J_0^2$ depending on the height above the photosphere")

    lambda_A = np.arange(4000, 12000, 1)
    for h_arcsec in [0, 3, 10, 40][::-1]:
        n = [radiation_tensor.n_fit(lam) for lam in lambda_A]
        w = [radiation_tensor.w_fit(lam, h_arcsec) for lam in lambda_A]
        ax[0].plot(lambda_A, n, label=rf"{h_arcsec=}")
        ax[1].plot(lambda_A, w, label=rf"{h_arcsec=}")

    ax[0].set_xlabel(r"$\lambda$ (Å)")
    ax[0].set_ylabel(r"$n$")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-7, 1e-1)

    ax[1].set_xlabel(r"$\lambda$ (Å)")
    ax[1].set_ylabel(r"$w$")
    ax[1].set_ylim(0, 0.4)
    ax[1].legend()

    transitions_to_plot = [
        "2p3_L=1.0_S=1.0->2s3_L=0.0_S=1.0_2.0_0.0",
        "3p3_L=1.0_S=1.0->2s3_L=0.0_S=1.0_2.0_0.0",
        "3s3_L=0.0_S=1.0->2p3_L=1.0_S=1.0_2.0_0.0",
        "3d3_L=2.0_S=1.0->2p3_L=1.0_S=1.0_2.0_0.0",
    ]
    transition_labels = [
        "2p3>2s3",
        "3p3>2s3",
        "3s3>2p3",
        "3d3>2p3",
    ]
    x_axis = range(len(transitions_to_plot))

    for h_arcsec in [40, 30, 20, 10, 5, 2, 1, 0]:
        logging.info("===========================")
        radiation_tensor.fill_NLTE_n_w_parametrized(h_arcsec=h_arcsec)
        ax[2].scatter(
            x_axis,
            [radiation_tensor.data[transition_id] for transition_id in transitions_to_plot],
            label=f"h_arcsec={h_arcsec}",
        )
    radiation_tensor.fill_planck(T_K=5700)
    ax[2].scatter(x_axis, [0 for _ in transitions_to_plot], label="LTE")

    ax[2].set_xlabel(r"Transition in He I atom")
    ax[2].set_ylabel(r"$J_0^2$ (erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$ Hz$^{-1}$)")
    ax[2].set_xticks(x_axis)
    ax[2].set_xticklabels(transition_labels)
    ax[2].tick_params(axis="x", labelrotation=90)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
