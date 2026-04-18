from enum import Enum, auto

import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import lambda_vacuum_to_air


class StokesNorm(Enum):
    """Normalization mode for Stokes profile plots."""

    NONE = auto()  # raw values, no normalization
    MAX_I = auto()  # divide all components by max(I)
    BY_REFERENCE = auto()  # divide all components by reference Stokes I (continuum)
    MAX_IpV_ImV = auto()  # normalize I+V and I-V each by their own max (IpmV plotter only)


def _x_axis(lambda_A, lambda_ref_A, vacuum_to_air):
    """Compute x-axis values and label from wavelength array."""
    if vacuum_to_air:
        x = lambda_vacuum_to_air(lambda_A)
        ref = lambda_vacuum_to_air(lambda_ref_A) if lambda_ref_A is not None else None
    else:
        x = lambda_A
        ref = lambda_ref_A
    if ref is not None:
        return x - ref, r"$\Delta\lambda_\mathrm{air}$ ($\AA$)" if vacuum_to_air else r"$\Delta\lambda$ ($\AA$)"
    return x, r"$\lambda_\mathrm{air}$ ($\AA$)" if vacuum_to_air else r"$\lambda$ ($\AA$)"


class StokesPlotter_IV:  # pragma: no cover
    """
    Stokes plotter class for Stokes I and V profiles.
    """

    def __init__(self, title="", vacuum_to_air=False):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.vacuum_to_air = vacuum_to_air
        self.fig, self.axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 8), num=title)
        self.fig.suptitle(title)
        self.axs[0].set_ylabel(r"Stokes $I/I_{max}$")
        self.axs[1].set_ylabel(r"Stokes $V/I_{max}$")
        self._x_label = None

    def add(self, lambda_A, stokes_I, stokes_V, lambda_ref_A=None, color=None, label="", linewidth=1.5):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        x, x_label = _x_axis(lambda_A, lambda_ref_A, self.vacuum_to_air)
        self._x_label = x_label

        if stokes_I is not None:
            self.axs[0].plot(x, stokes_I / np.max(stokes_I), label=label, color=color, linewidth=linewidth)
        if stokes_V is not None:
            self.axs[1].plot(x, stokes_V / np.max(stokes_I), label=label, color=color, linewidth=linewidth)

    def add_stokes(
        self,
        lambda_A,
        stokes: Stokes,
        lambda_ref_A=None,
        stokes_reference: Stokes = None,
        norm: StokesNorm = StokesNorm.NONE,
        color=None,
        label="",
        linewidth=1.5,
    ):
        if norm == StokesNorm.MAX_I:
            scale = np.max(stokes.I)
        elif norm == StokesNorm.BY_REFERENCE:
            scale = stokes_reference.I
        else:
            scale = 1
        self.add(
            lambda_A=lambda_A,
            lambda_ref_A=lambda_ref_A,
            stokes_I=stokes.I / scale,
            stokes_V=stokes.V / scale,
            color=color,
            label=label,
            linewidth=linewidth,
        )

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[1].set_xlabel(self._x_label or r"$\lambda$ ($\AA$)")

        for ax in self.axs:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize="x-small",
            )

        plt.show()


class StokesPlotter_IV_IpmV:  # pragma: no cover
    r"""
    Stokes plotter class for Stokes :math:`I, V, I\pm V` profiles.
    """

    def __init__(self, title="", vacuum_to_air=False):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.vacuum_to_air = vacuum_to_air
        self.fig, self.axs = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(8, 8), num=title)
        self.fig.suptitle(title)
        self.axs[0].set_ylabel(r"Stokes $I$")
        self.axs[1].set_ylabel(r"Stokes $V$")
        self.axs[2].set_ylabel(r"Stokes $(I\pm V)$")
        self._x_label = None

    def add(self, lambda_A, stokes_I, stokes_V, lambda_ref_A=None, color=None, label="", linewidth=1.5):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        x, x_label = _x_axis(lambda_A, lambda_ref_A, self.vacuum_to_air)
        self._x_label = x_label

        self.axs[0].plot(x, stokes_I, label=label, color=color, linewidth=linewidth)
        self.axs[1].plot(x, stokes_V, label=label, color=color, linewidth=linewidth)
        self.axs[2].plot(x, stokes_I + stokes_V, "-", label=label + " $I+V$", color=color, linewidth=linewidth)
        self.axs[2].plot(x, stokes_I - stokes_V, "--", label=label + " $I-V$", color=color, linewidth=linewidth)

    def add_stokes(
        self,
        lambda_A,
        stokes: Stokes,
        lambda_ref_A=None,
        stokes_reference: Stokes = None,
        norm: StokesNorm = StokesNorm.NONE,
        color=None,
        label="",
        linewidth=1.5,
    ):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        x, x_label = _x_axis(lambda_A, lambda_ref_A, self.vacuum_to_air)
        self._x_label = x_label

        if norm == StokesNorm.NONE:
            I, V = stokes.I, stokes.V
        elif norm == StokesNorm.MAX_I:
            scale = np.max(stokes.I)
            I, V = stokes.I / scale, stokes.V / scale
        elif norm == StokesNorm.BY_REFERENCE:
            scale = stokes_reference.I
            I, V = stokes.I / scale, stokes.V / scale
        elif norm == StokesNorm.MAX_IpV_ImV:
            I, V = stokes.I, stokes.V

        self.axs[0].plot(x, I, label=label, color=color, linewidth=linewidth)
        self.axs[1].plot(x, V, label=label, color=color, linewidth=linewidth)

        if norm == StokesNorm.MAX_IpV_ImV:
            IpV = stokes.I + stokes.V
            ImV = stokes.I - stokes.V
            IpV = IpV / np.max(np.abs(IpV))
            ImV = ImV / np.max(np.abs(ImV))
        else:
            IpV = I + V
            ImV = I - V

        self.axs[2].plot(x, IpV, "-", label=label + " $I+V$", color=color, linewidth=linewidth)
        self.axs[2].plot(x, ImV, "--", label=label + " $I-V$", color=color, linewidth=linewidth)

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[2].grid(True)
        self.axs[2].set_xlabel(self._x_label or r"$\lambda$ ($\AA$)")

        for ax in self.axs:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize="x-small",
            )

        plt.show()


class StokesPlotter:  # pragma: no cover
    r"""
    Stokes plotter class for Stokes :math:`I, Q, U, V` profiles.
    """

    def __init__(
        self,
        title="",
        vacuum_to_air=False,
        x_label=None,
        y_label_I=r"Stokes $I$",
        y_label_Q=r"Stokes $Q$",
        y_label_U=r"Stokes $U$",
        y_label_V=r"Stokes $V$",
    ):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.vacuum_to_air = vacuum_to_air
        self.fig, self.axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=(8, 8), num=title)
        self.fig.suptitle(title)
        self.axs[0].set_ylabel(y_label_I)
        self.axs[1].set_ylabel(y_label_Q)
        self.axs[2].set_ylabel(y_label_U)
        self.axs[3].set_ylabel(y_label_V)
        self._x_label = x_label  # explicit override (e.g. for frequency axis)

    def add(
        self,
        lambda_A,
        stokes_I,
        stokes_Q,
        stokes_U,
        stokes_V,
        lambda_ref_A=None,
        color=None,
        label="",
        style="-",
        linewidth=1.5,
    ):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        x, x_label = _x_axis(lambda_A, lambda_ref_A, self.vacuum_to_air)
        if self._x_label is None:
            self._x_label = x_label

        if stokes_I is not None:
            self.axs[0].plot(x, stokes_I, style, label=label, color=color, linewidth=linewidth)
        if stokes_Q is not None:
            self.axs[1].plot(x, stokes_Q, style, label=label, color=color, linewidth=linewidth)
        if stokes_U is not None:
            self.axs[2].plot(x, stokes_U, style, label=label, color=color, linewidth=linewidth)
        if stokes_V is not None:
            self.axs[3].plot(x, stokes_V, style, label=label, color=color, linewidth=linewidth)

    def add_stokes(
        self,
        lambda_A,
        stokes: Stokes,
        lambda_ref_A=None,
        stokes_reference: Stokes = None,
        norm: StokesNorm = StokesNorm.NONE,
        color=None,
        label="",
        style="-",
        linewidth=1.5,
    ):
        if norm == StokesNorm.MAX_I:
            scale = np.max(stokes.I)
        elif norm == StokesNorm.BY_REFERENCE:
            scale = stokes_reference.I
        else:
            scale = 1
        self.add(
            lambda_A=lambda_A,
            lambda_ref_A=lambda_ref_A,
            stokes_I=stokes.I / scale,
            stokes_Q=stokes.Q / scale,
            stokes_U=stokes.U / scale,
            stokes_V=stokes.V / scale,
            color=color,
            label=label,
            style=style,
            linewidth=linewidth,
        )

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[2].grid(True)
        self.axs[3].grid(True)
        self.axs[3].set_xlabel(self._x_label or r"$\lambda$ ($\AA$)")

        for ax in self.axs:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize="x-small",
            )

        plt.show()
