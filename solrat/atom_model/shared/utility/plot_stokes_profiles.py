from enum import Enum, auto

import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    frequency_sm1_to_lambda_A,
    lambda_air_to_vacuum,
    lambda_vacuum_to_air,
)
from solrat.engine.functions.decorators import log_method


class StokesNorm(Enum):
    """Normalization mode for Stokes profile plots."""

    NONE = auto()  # raw values, no normalization
    MAX_I = auto()  # divide all components by max(I)
    BY_REFERENCE = auto()  # divide all components by reference Stokes I (continuum)
    MAX_IpV_ImV = auto()  # normalize I+V and I-V each by their own max (IpmV plotter only)


def _compute_wavelength_axis(nu, reference_lambda_A_air, use_air_wavelengths):
    lambda_A_vac = frequency_sm1_to_lambda_A(nu)
    if use_air_wavelengths:
        wavelength = lambda_vacuum_to_air(lambda_A_vac)
        ref = reference_lambda_A_air
    else:
        wavelength = lambda_A_vac
        ref = lambda_air_to_vacuum(reference_lambda_A_air) if reference_lambda_A_air is not None else None
    if ref is not None:
        label = (
            r"$\Delta\lambda_\mathrm{air}$ ($\AA$)" if use_air_wavelengths else r"$\Delta\lambda_\mathrm{vac}$ ($\AA$)"
        )
        return wavelength - ref, label
    label = r"$\lambda_\mathrm{air}$ ($\AA$)" if use_air_wavelengths else r"$\lambda_\mathrm{vac}$ ($\AA$)"
    return wavelength, label


class PlotterBase:  # pragma: no cover
    Norm = StokesNorm

    def __init__(
        self, title, use_air_wavelengths, reference_lambda_A_air, n_axes, y_labels, figsize=(8, 8), x_label=None
    ):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.vacuum_to_air = use_air_wavelengths
        self.reference_lambda_A_air = reference_lambda_A_air
        self.fig, self.axs = plt.subplots(n_axes, 1, sharex=True, constrained_layout=True, figsize=figsize, num=title)
        if n_axes == 1:
            self.axs = [self.axs]
        for ax, label in zip(self.axs, y_labels):
            ax.set_ylabel(label)
        self._wavelength_label = x_label

    def _next_color(self, color):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1
        return color

    def _wavelength_axis(self, nu):
        wavelength, label = _compute_wavelength_axis(nu, self.reference_lambda_A_air, self.vacuum_to_air)
        if self._wavelength_label is None:
            self._wavelength_label = label
        return wavelength

    @log_method
    def finalize(self, legend=True):
        for ax in self.axs:
            ax.grid(True)
            if legend:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize="x-small")
        self.axs[-1].set_xlabel(self._wavelength_label or r"$\lambda_\mathrm{vac}$ ($\AA$)")

    @log_method
    def show(self):
        self.finalize()
        plt.show()


class StokesPlotter_IV(PlotterBase):  # pragma: no cover
    """
    Stokes plotter class for Stokes I and V profiles.
    """

    def __init__(self, title="", use_air_wavelengths=True, reference_lambda_A_air=None):
        super().__init__(
            title=title,
            use_air_wavelengths=use_air_wavelengths,
            reference_lambda_A_air=reference_lambda_A_air,
            n_axes=2,
            y_labels=[r"Stokes $I/I_{max}$", r"Stokes $V/I_{max}$"],
        )

    @log_method
    def add(self, nu, stokes_I, stokes_V, color=None, label="", linewidth=1.5):
        color = self._next_color(color)
        wavelength = self._wavelength_axis(nu)
        if stokes_I is not None:
            self.axs[0].plot(wavelength, stokes_I / np.max(stokes_I), label=label, color=color, linewidth=linewidth)
        if stokes_V is not None:
            self.axs[1].plot(wavelength, stokes_V / np.max(stokes_I), label=label, color=color, linewidth=linewidth)

    @log_method
    def add_stokes(
        self,
        nu,
        stokes: Stokes,
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
        elif norm == StokesNorm.NONE:
            scale = 1
        else:
            raise ValueError(f"Did not recognize normalization option {norm}")

        self.add(
            nu=nu,
            stokes_I=stokes.I / scale,
            stokes_V=stokes.V / scale,
            color=color,
            label=label,
            linewidth=linewidth,
        )


class StokesPlotter_IV_IpmV(PlotterBase):  # pragma: no cover
    r"""
    Stokes plotter class for Stokes :math:`I, V, I\pm V` profiles.
    """

    def __init__(self, title="", use_air_wavelengths=False, reference_lambda_A_air=None):
        super().__init__(
            title=title,
            use_air_wavelengths=use_air_wavelengths,
            reference_lambda_A_air=reference_lambda_A_air,
            n_axes=3,
            y_labels=[r"Stokes $I$", r"Stokes $V$", r"Stokes $(I\pm V)$"],
        )

    @log_method
    def add(self, nu, stokes_I, stokes_V, color=None, label="", linewidth=1.5):
        color = self._next_color(color)
        wavelength = self._wavelength_axis(nu)
        self.axs[0].plot(wavelength, stokes_I, label=label, color=color, linewidth=linewidth)
        self.axs[1].plot(wavelength, stokes_V, label=label, color=color, linewidth=linewidth)
        self.axs[2].plot(wavelength, stokes_I + stokes_V, "-", label=label + " $I+V$", color=color, linewidth=linewidth)
        self.axs[2].plot(
            wavelength, stokes_I - stokes_V, "--", label=label + " $I-V$", color=color, linewidth=linewidth
        )

    @log_method
    def add_stokes(
        self,
        nu,
        stokes: Stokes,
        stokes_reference: Stokes = None,
        norm: StokesNorm = StokesNorm.NONE,
        color=None,
        label="",
        linewidth=1.5,
    ):
        color = self._next_color(color)
        wavelength = self._wavelength_axis(nu)

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
        else:
            raise ValueError(f"Did not recognize normalization option {norm}")

        self.axs[0].plot(wavelength, I, label=label, color=color, linewidth=linewidth)
        self.axs[1].plot(wavelength, V, label=label, color=color, linewidth=linewidth)

        if norm == StokesNorm.MAX_IpV_ImV:
            IpV = stokes.I + stokes.V
            ImV = stokes.I - stokes.V
            IpV = IpV / np.max(np.abs(IpV))
            ImV = ImV / np.max(np.abs(ImV))
        else:
            IpV = I + V
            ImV = I - V

        self.axs[2].plot(wavelength, IpV, "-", label=label + " $I+V$", color=color, linewidth=linewidth)
        self.axs[2].plot(wavelength, ImV, "--", label=label + " $I-V$", color=color, linewidth=linewidth)


class StokesPlotter_IpmV(PlotterBase):  # pragma: no cover
    r"""
    Stokes plotter class for Stokes :math:`I\pm V` profiles.
    """

    def __init__(self, title="", use_air_wavelengths=False, reference_lambda_A_air=None, figsize=(8, 6)):
        super().__init__(
            title=title,
            use_air_wavelengths=use_air_wavelengths,
            reference_lambda_A_air=reference_lambda_A_air,
            n_axes=1,
            y_labels=[r"Stokes $(I\pm V)$"],
            figsize=figsize,
        )

    @log_method
    def add_stokes(
        self,
        stokes: Stokes,
        stokes_reference: Stokes = None,
        norm: StokesNorm = StokesNorm.NONE,
        label="",
        linewidth=1.5,
        alpha=1,
    ):
        wavelength = self._wavelength_axis(stokes.nu)

        if norm == StokesNorm.NONE:
            I, V = stokes.I, stokes.V
        elif norm == StokesNorm.MAX_I:
            scale = np.max(stokes.I)
            I, V = stokes.I / scale, stokes.V / scale
        elif norm == StokesNorm.BY_REFERENCE:
            scale = stokes_reference.I
            I, V = stokes.I / scale, stokes.V / scale
        elif norm == StokesNorm.MAX_IpV_ImV:
            scale = np.max(stokes.I)
            I, V = stokes.I / scale, stokes.V / scale
        else:
            raise ValueError(f"Did not recognize normalization option {norm}")

        if norm == StokesNorm.MAX_IpV_ImV:
            IpV = stokes.I + stokes.V
            ImV = stokes.I - stokes.V
            IpV = IpV / np.max(np.abs(IpV))
            ImV = ImV / np.max(np.abs(ImV))
        else:
            IpV = I + V
            ImV = I - V

        self.axs[0].plot(wavelength, IpV, "-", label=label + " $I+V$", color="blue", linewidth=linewidth, alpha=alpha)
        self.axs[0].plot(wavelength, ImV, "-", label=label + " $I-V$", color="red", linewidth=linewidth, alpha=alpha)


class StokesPlotter(PlotterBase):  # pragma: no cover
    r"""
    Stokes plotter class for Stokes :math:`I, Q, U, V` profiles.
    """

    def __init__(
        self,
        title="",
        use_air_wavelengths=False,
        reference_lambda_A_air=None,
        x_label=None,
        y_label_I=r"Stokes $I$",
        y_label_Q=r"Stokes $Q$",
        y_label_U=r"Stokes $U$",
        y_label_V=r"Stokes $V$",
    ):
        super().__init__(
            title=title,
            use_air_wavelengths=use_air_wavelengths,
            reference_lambda_A_air=reference_lambda_A_air,
            n_axes=4,
            y_labels=[y_label_I, y_label_Q, y_label_U, y_label_V],
            x_label=x_label,
        )

    @log_method
    def add(
        self,
        nu,
        stokes_I,
        stokes_Q,
        stokes_U,
        stokes_V,
        color=None,
        label="",
        style="-",
        linewidth=1.5,
    ):
        color = self._next_color(color)
        wavelength = self._wavelength_axis(nu)

        if stokes_I is not None:
            self.axs[0].plot(wavelength, stokes_I, style, label=label, color=color, linewidth=linewidth)
        if stokes_Q is not None:
            self.axs[1].plot(wavelength, stokes_Q, style, label=label, color=color, linewidth=linewidth)
        if stokes_U is not None:
            self.axs[2].plot(wavelength, stokes_U, style, label=label, color=color, linewidth=linewidth)
        if stokes_V is not None:
            self.axs[3].plot(wavelength, stokes_V, style, label=label, color=color, linewidth=linewidth)

    @log_method
    def add_stokes(
        self,
        nu,
        stokes: Stokes,
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
        elif norm == StokesNorm.NONE:
            scale = 1
        else:
            raise ValueError(f"Did not recognize normalization option {norm}")
        self.add(
            nu=nu,
            stokes_I=stokes.I / scale,
            stokes_Q=stokes.Q / scale,
            stokes_U=stokes.U / scale,
            stokes_V=stokes.V / scale,
            color=color,
            label=label,
            style=style,
            linewidth=linewidth,
        )
