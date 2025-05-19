import numpy as np
from matplotlib import pyplot as plt


class StokesPlotterTwoPanel:
    def __init__(self, title=""):
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.next_color_index = 0
        self.n_items = 0
        self.fig, self.axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(6, 10), num=title)
        # self.axs[0].set_title(title)
        self.axs[0].set_ylabel(r"Stokes $I/I_{max}$, $V/I_{max}$")
        self.axs[1].set_ylabel(r"Stokes $Q/I_{max}$, $U/I_{max}$")

    def add(self, lambda_A, reference_lambda_A, stokes_I, stokes_Q, stokes_U, stokes_V, color=None, label=""):
        if color == "auto":
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        if stokes_I is not None:
            self.axs[0].plot(
                lambda_A - reference_lambda_A, stokes_I / max(stokes_I), label=r"$I$ " + label, color=color, linewidth=1
            )
        if stokes_V is not None:
            self.axs[0].plot(
                lambda_A - reference_lambda_A,
                stokes_V / max(stokes_I),
                ":",
                label=r"$V$ " + label,
                color=color,
                linewidth=1.5,
            )
        if stokes_Q is not None:
            self.axs[1].plot(
                lambda_A - reference_lambda_A, stokes_Q / max(stokes_I), label=r"$Q$ " + label, color=color, linewidth=1
            )
        if stokes_U is not None:
            self.axs[1].plot(
                lambda_A - reference_lambda_A,
                stokes_U / max(stokes_I),
                ":",
                label=r"$U$ " + label,
                color=color,
                linewidth=1.5,
            )
        self.n_items += 1

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[1].set_xlabel(r"$\Delta\lambda$ ($\AA$)")
        # handles, labels = self.axs[0].get_legend_handles_labels()
        # handles = np.concatenate((handles[::2], handles[1::2]), axis=0)
        # labels = np.concatenate((labels[::2], labels[1::2]), axis=0)

        self.axs[0].legend(
            # handles,
            # labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),  # Centered above the axes
            ncol=self.n_items,  # 2 columns -> 2 rows with 4 items
            fontsize="x-small",
            # draggable=True,
        )

        self.axs[1].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),  # Centered above the axes
            ncol=self.n_items,  # 2 columns -> 2 rows with 4 items
            fontsize="x-small",
            # draggable=True,
        )
        # plt.tight_layout()
        plt.show()
