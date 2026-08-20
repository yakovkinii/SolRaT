import matplotlib.pyplot as plt
import numpy as np

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    frequencies_around_line_sm1,
    height_grid_refined_at_observer_surface,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging


def main():
    r"""
    Compare the depth-resolution convergence of the first-order (``delo_constant``) and second-order
    (``delo_linear``) DELO transfer schemes on the TB1999 (:math:`\mu=0.1`) scattering line: the
    emergent line-center :math:`Q/I` error against the digitized TB1999 benchmark versus the number
    of surface depth points.
    """
    setup_logging()

    temperature_K = 6000.0

    collisions = ParametrizedCollisions()
    model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
    transition = next(iter(model.config.transition_registry.transitions.values()))
    collisions.set_deexcitation_rate_from_epsilon(transition=transition, epsilon=1e-2, temperature_K=temperature_K)

    params = model.AtmosphereParameters(
        model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
    )
    nu0 = transition.get_mean_transition_frequency_sm1()
    nu = frequencies_around_line_sm1(nu0, params.delta_v_thermal_cm_sm1, step_doppler=0.1)
    line_center = int(np.argmin(np.abs(nu - nu0)))
    state = None

    def qi(n_near_surface, transfer_scheme):
        nonlocal state
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=height_grid_refined_at_observer_surface(1000e5, n_near_surface, n_near_surface // 2),
            temperature_K=temperature_K,
            number_density_cm3=1.0e11,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=float(np.arccos(0.1)),
            n_mu_quadrature=4,
            n_phi_quadrature=3,
            max_iterations=2000,
            tolerance=1e-8,
            ng_acceleration=True,
            ng_damping=0.5,
            ng_period=8,
            transfer_scheme=transfer_scheme,
            estimate_true_error=True,
        )
        emergent = atmosphere.forward(
            initial_stokes=Stokes.from_zeros(nu_sm1=nu),
            initial_state=state if state is not None else None,
        )
        state = atmosphere.get_state()
        assert atmosphere.iterations_used < atmosphere.max_iterations, (
            f"{transfer_scheme} at N={n_near_surface} did not reach the true-error tolerance in "
            f"{atmosphere.max_iterations} iterations (estimate {atmosphere.final_true_error})."
        )
        return 100.0 * emergent.Q[line_center] / emergent.I[line_center]

    node_counts = np.array([2, 5, 10, 20, 50, 100])
    tb1999_line_center_qi = -3.40158  # TB1999 Fig. 10 (mu=0.1) line-center 100 Q/I
    errors = {
        scheme: np.array([abs(qi(n, scheme) - tb1999_line_center_qi) for n in node_counts])
        for scheme in ("delo_constant", "delo_linear")
    }
    slopes = {scheme: np.polyfit(np.log(node_counts), np.log(errors[scheme]), 1)[0] for scheme in errors}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(node_counts, errors["delo_constant"], marker="o", color="k", label="DELO-constant (1st order)")
    ax.loglog(node_counts, errors["delo_linear"], marker="s", color="#d62728", label="DELO-linear (2nd order)")
    ax.set_xlabel("surface depth points $N$")
    ax.set_ylabel(r"$|100\,Q/I - \mathrm{TB1999}|$ at line center")
    ax.legend()
    fig.tight_layout()

    print(
        f"DELO scheme depth-convergence (TB1999 mu=0.1): TB1999 100 Q/I = {tb1999_line_center_qi:.4f}; "
        f"fitted order (log-log slope) constant = {-slopes['delo_constant']:.2f}, "
        f"linear = {-slopes['delo_linear']:.2f}"
    )
    return fig


if __name__ == "__main__":
    main()
    plt.show()
