from matplotlib import pyplot as plt


class StokesPlotterTwoPanel:
    def __init__(self, title=''):
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.next_color_index = 0
        self.fig, self.axs = plt.subplots(2, 1, sharex=True)
        self.axs[0].set_title(title)
        self.axs[0].set_ylabel(r"Stokes $I/I_{max}$, $V/I_{max}$")
        self.axs[1].set_ylabel(r"Stokes $Q/I_{max}$, $U/I_{max}$")

    def add(self, lambda_A, reference_lambda_A, stokes_I, stokes_Q, stokes_U, stokes_V, color=None, label=""):
        if color == 'auto':
            color = self.colors[self.next_color_index % len(self.colors)]
            self.next_color_index += 1

        self.axs[0].plot(lambda_A - reference_lambda_A, stokes_I / max(stokes_I), label=r"$I$ " + label, color=color, linewidth=1)
        self.axs[0].plot(lambda_A - reference_lambda_A, stokes_V / max(stokes_I),':', label=r"$V$ " + label, color=color, linewidth=1.5)
        self.axs[1].plot(lambda_A - reference_lambda_A, stokes_Q / max(stokes_I), label=r"$Q$ " + label, color=color, linewidth=1)
        self.axs[1].plot(lambda_A - reference_lambda_A, stokes_U / max(stokes_I),':', label=r"$U$ " + label, color=color, linewidth=1.5)

    def show(self):
        self.axs[0].grid(True)
        self.axs[1].grid(True)
        self.axs[1].set_xlabel(r"$\Delta\lambda$ ($\AA$)")
        self.axs[0].legend()
        self.axs[1].legend()
        # plt.tight_layout()
        plt.show()