"""
TODO
TODO  This file needs improved documentation.
TODO
"""

import numpy as np
from matplotlib import pyplot as plt

from src.common.functions import frequency_hz_to_lambda_A
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes


def radiation_tensor_NLTE_n_w_parametrized(
    multi_term_atom_context: MultiTermAtomContext, h_arcsec: float
) -> RadiationTensor:
    """
    Create a parametrized NLTE radiation tensor for given height above limb.

    Args:
        multi_term_atom_context: Atomic model context
        h_arcsec: Height above limb in arcseconds

    Returns:
        Configured RadiationTensor
    """
    return RadiationTensor(transition_registry=multi_term_atom_context.transition_registry).fill_NLTE_n_w_parametrized(
        h_arcsec=h_arcsec
    )


def plot_stokes_IQUV(
    stokes: Stokes, label: str, reference_lambda_A: float, show: bool = True, axs=None, normalize=True, color=None
):
    """
    Plot all four Stokes parameters (I, Q, U, V) vs wavelength.

    Args:
        stokes: Stokes vector to plot
        label: Label for the plot legend
        reference_lambda_A: Reference wavelength for x-axis offset
        show: Whether to show the plot immediately
        axs: Existing axes to plot on (if None, creates new figure)
        normalize: Whether to normalize by peak intensity
        color: Color for the lines (if None, uses default)

    Returns:
        fig, axs: Figure and axes objects
    """
    if axs is None:
        fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=(10, 8))
    else:
        fig = axs[0].figure

    norm = 1.0
    if normalize:
        norm = max(stokes.I)

    lambda_A = frequency_hz_to_lambda_A(stokes.nu)
    delta_lambda = lambda_A - reference_lambda_A

    plot_kwargs = {"label": f"I ({label})"}
    if color is not None:
        plot_kwargs["color"] = color

    axs[0].plot(delta_lambda, stokes.I / norm, **plot_kwargs)
    axs[0].set_ylabel("I")
    axs[0].grid(True)

    plot_kwargs["label"] = f"Q ({label})"
    axs[1].plot(delta_lambda, stokes.Q / norm, **plot_kwargs)
    axs[1].set_ylabel("Q")
    axs[1].grid(True)

    plot_kwargs["label"] = f"U ({label})"
    axs[2].plot(delta_lambda, stokes.U / norm, **plot_kwargs)
    axs[2].set_ylabel("U")
    axs[2].grid(True)

    plot_kwargs["label"] = f"V ({label})"
    axs[3].plot(delta_lambda, stokes.V / norm, **plot_kwargs)
    axs[3].set_ylabel("V")
    axs[3].set_xlabel(r"$\Delta\lambda$ ($\AA$)")
    axs[3].grid(True)

    for ax in axs:
        ax.legend(loc="best", fontsize="x-small")

    if show:
        plt.show()

    return fig, axs


def plot_stokes_comparison(
    stokes_list, labels, reference_lambda_A: float, title="Stokes Parameter Comparison", colors=None
):
    """
    Plot multiple Stokes vectors for comparison.

    Args:
        stokes_list: List of Stokes objects
        labels: List of labels for each Stokes vector
        reference_lambda_A: Reference wavelength
        title: Plot title
        colors: List of colors (optional)

    Returns:
        fig, axs: Figure and axes objects
    """
    fig, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    if colors is None:
        colors = [None] * len(stokes_list)

    for stokes, label, color in zip(stokes_list, labels, colors):
        plot_stokes_IQUV(stokes, label, reference_lambda_A, show=False, axs=axs, normalize=True, color=color)

    plt.show()
    return fig, axs
