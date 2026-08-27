import time

import numpy as np
from matplotlib import pyplot as plt

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.shared.common_api.constant_property_slab import ConstantPropertySlabAtmosphere
from solrat.atom_model.shared.common_api.multi_slab_atmosphere import MultiSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range
from solrat.atom_model.shared.utility.log_setup import setup_logging


def time_forward(model, radiation_tensor, atmosphere_parameters, angles, nu: np.ndarray, repeats: int) -> float:
    r"""
    Median wall-clock time of a full constant-property-slab forward synthesis (SEE solution plus the
    per-frequency radiative transfer) on the given frequency grid.

    :param model: configured Model.
    :param radiation_tensor: prescribed radiation tensor.
    :param atmosphere_parameters: atmosphere parameters.
    :param angles: observation and field geometry.
    :param nu: frequency grid [1/s].
    :param repeats: number of timed repetitions.
    :return: median forward-synthesis time [s].
    """
    atmosphere = MultiSlabAtmosphere(
        ConstantPropertySlabAtmosphere(
            model=model,
            radiation_tensor=radiation_tensor,
            line_delta_tau=0.1,
            continuum_delta_tau=1.0e-3,
            angles=angles,
            atmosphere_parameters=atmosphere_parameters,
        )
    )
    initial_stokes = Stokes.from_zeros(nu_sm1=nu)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        atmosphere.forward(initial_stokes=initial_stokes)
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def main():
    r"""
    Forward-synthesis time versus number of frequency points, for a He I D3 constant-property slab.

    :return: matplotlib Figure.
    """
    setup_logging()

    model = PreconfiguredModels.multi_term_atom_HeID3()
    reference_lambda_A_air = model.config.reference_lambda_A_air
    angles = Angles(chi=0.0, theta=np.pi / 4, gamma=0.0, chi_B=0.0, theta_B=np.pi / 6)
    atmosphere_parameters = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=1000.0, temperature_K=7000.0
    )
    radiation_tensor = model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=30)

    n_frequency_points = [50, 100, 200, 400, 800, 1600, 3200]
    forward_times = []
    for n_points in n_frequency_points:
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.6,
            upper_wavelength_A=reference_lambda_A_air + 0.6,
            step_A=1.2 / n_points,
        )
        forward_times.append(time_forward(model, radiation_tensor, atmosphere_parameters, angles, nu, repeats=3))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n_frequency_points, np.array(forward_times) * 1e3, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of frequency points")
    ax.set_ylabel("forward-synthesis time (ms)")
    ax.set_title("He I D3 constant-property slab: forward-synthesis scaling")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    scaling = "  ".join(f"{n}:{t * 1e3:.1f}ms" for n, t in zip(n_frequency_points, forward_times))
    print(f"Forward-synthesis scaling (He I D3 slab): {scaling}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
