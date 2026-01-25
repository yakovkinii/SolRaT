"""
TODO
TODO  This file needs improved documentation.
TODO
"""

from matplotlib import pyplot as plt

from src.multi_term_atom.object.stokes import Stokes


class StokesPlotter_IV:
    def __init__(self, title=""):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.fig, self.axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 8), num=title)
        self.axs[0].set_ylabel(r"Stokes $I/I_{max}$")
        self.axs[1].set_ylabel(r"Stokes $V/I_{max}$")

    def add(self, lambda_A, reference_lambda_A, stokes_I, stokes_V, color=None, label=""):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        if stokes_I is not None:
            self.axs[0].plot(
                lambda_A - reference_lambda_A, stokes_I / max(stokes_I), label=label, color=color, linewidth=1
            )
        if stokes_V is not None:
            self.axs[1].plot(
                lambda_A - reference_lambda_A, stokes_V / max(stokes_I), label=label, color=color, linewidth=1
            )

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[1].set_xlabel(r"$\Delta\lambda$ ($\AA$)")

        self.axs[0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Centered above the axes
            fontsize="x-small",
        )

        self.axs[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Centered above the axes
            fontsize="x-small",
        )
        plt.show()


class StokesPlotter_IV_IpmV:
    def __init__(self, title=""):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.fig, self.axs = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(8, 8), num=title)
        self.axs[0].set_ylabel(r"Stokes $I$")
        self.axs[1].set_ylabel(r"Stokes $V$")
        self.axs[2].set_ylabel(r"Stokes $(I\pm V)$")

    def add(self, lambda_A, reference_lambda_A, stokes_I, stokes_V, color=None, label=""):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        self.axs[0].plot(lambda_A - reference_lambda_A, stokes_I, label=label, color=color, linewidth=1)
        self.axs[1].plot(lambda_A - reference_lambda_A, stokes_V, label=label, color=color, linewidth=1)
        self.axs[2].plot(
            lambda_A - reference_lambda_A,
            (stokes_I + stokes_V),
            "-",
            label=label + " $I+V$",
            color=color,
            linewidth=1,
        )
        self.axs[2].plot(
            lambda_A - reference_lambda_A,
            (stokes_I - stokes_V),
            "--",
            label=label + " $I-V$",
            color=color,
            linewidth=1,
        )

    def add_stokes(
        self,
        lambda_A,
        reference_lambda_A,
        stokes: Stokes,
        stokes_reference: Stokes = None,
        color=None,
        label="",
    ):
        scale = 1 if stokes_reference is None else stokes_reference.I
        self.add(
            lambda_A=lambda_A,
            reference_lambda_A=reference_lambda_A,
            stokes_I=stokes.I / scale,
            stokes_V=stokes.V / scale,
            color=color,
            label=label,
        )

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[2].grid(True)
        self.axs[2].set_xlabel(r"$\Delta\lambda$ ($\AA$)")

        self.axs[0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Centered above the axes
            fontsize="x-small",
        )

        self.axs[1].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Centered above the axes
            fontsize="x-small",
        )
        self.axs[2].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Centered above the axes
            fontsize="x-small",
        )
        plt.show()


class StokesPlotter:
    def __init__(
        self,
        title="",
        x_label=r"$\Delta\lambda$ ($\AA$)",
        y_label_I=r"Stokes $I$",
        y_label_Q=r"Stokes $Q$",
        y_label_U=r"Stokes $U$",
        y_label_V=r"Stokes $V$",
    ):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.fig, self.axs = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=(8, 8), num=title)
        self.axs[0].set_ylabel(y_label_I)
        self.axs[1].set_ylabel(y_label_Q)
        self.axs[2].set_ylabel(y_label_U)
        self.axs[3].set_ylabel(y_label_V)
        self.x_label = x_label

    def add(
        self,
        lambda_A,
        reference_lambda_A,
        stokes_I,
        stokes_Q,
        stokes_U,
        stokes_V,
        color=None,
        label="",
        style="-",
        linewidth=1.5,
    ):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        if stokes_I is not None:
            self.axs[0].plot(
                lambda_A - reference_lambda_A, stokes_I, style, label=label, color=color, linewidth=linewidth
            )

        if stokes_Q is not None:
            self.axs[1].plot(
                lambda_A - reference_lambda_A, stokes_Q, style, label=label, color=color, linewidth=linewidth
            )

        if stokes_U is not None:
            self.axs[2].plot(
                lambda_A - reference_lambda_A, stokes_U, style, label=label, color=color, linewidth=linewidth
            )

        if stokes_V is not None:
            self.axs[3].plot(
                lambda_A - reference_lambda_A, stokes_V, style, label=label, color=color, linewidth=linewidth
            )

    def add_stokes(
        self,
        lambda_A,
        reference_lambda_A,
        stokes: Stokes,
        stokes_reference: Stokes = None,
        color=None,
        label="",
        style="-",
        linewidth=1.5,
    ):
        scale = 1 if stokes_reference is None else stokes_reference.I
        self.add(
            lambda_A=lambda_A,
            reference_lambda_A=reference_lambda_A,
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
        self.axs[3].set_xlabel(self.x_label)

        self.axs[0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),  # Centered above the axes
            fontsize="x-small",
        )

        plt.show()
