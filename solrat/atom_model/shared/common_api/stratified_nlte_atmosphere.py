import copy
import logging
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy import real

from solrat.atom_model.base_atom_model.object.radiation_tensor import BaseRadiationTensor
from solrat.atom_model.base_atom_model.object.rho import BaseRho
from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.model_registry import Model
from solrat.atom_model.shared.common_api.nlte_state import NLTEState
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.rotations import T_K_Q
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import get_planck_BP
from solrat.atom_model.shared.utility.voigt_profile import voigt
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.looping import FROMTO, PROJECTION
from solrat.engine.generators.nested_loops import nested_loops

# Each profile may be supplied as a scalar (constant with height), an array sampled on the
# height grid, or a callable f(z_cm) -> value.
Profile = Union[float, Sequence[float], np.ndarray, Callable[[float], float]]

# Static assert message for the per-ray transfer (hot path): no per-call string building.
_ERR_TANGENTIAL_MU = "Quadrature mu is too close to tangential (|mu| < 1e-6)."

# Observer |mu| below this is treated as an exactly tangential (mu = 0) line of sight: the emergent
# Stokes is taken as the surface source function (Eddington-Barbier limit I(0, mu->0) = S(tau=0)),
# so the diverging tangential path length is never integrated.
_MU_TANGENTIAL_THRESHOLD = 1e-4

# True-error estimation (opt-in): the residual r_k = max|Delta rho| is a step size, not the distance
# to the fixed point; with contraction rate lambda the error is ~ r_k / (1 - lambda). lambda is read
# from the geometric decay r_k / r_{k-1}. With Ng acceleration, only the "measure" iterations of each
# period decay geometrically: [jump] - [relax] - [measure] - [jump]. These set that window.
_NG_RELAX_ITERATIONS = 3  # iterations after an Ng jump discarded before the decay is clean
_MIN_MEASURE_ITERATIONS = 3  # clean residuals needed for a rate estimate (>= 2 ratios)
_LAMBDA_WINDOW = 8  # cap on residuals kept for the rate estimate


def _sample_profile(value: Profile, z_cm: np.ndarray, name: str) -> np.ndarray:
    r"""
    Sample a scalar / array / callable profile onto the height grid ``z_cm``.
    """
    if callable(value):
        return np.array([float(value(float(zi))) for zi in z_cm], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(len(z_cm), float(arr))
    msg = f"Profile '{name}' has length {arr.shape} but the height grid has length {z_cm.shape}."
    assert arr.shape == z_cm.shape, msg
    return arr


class StratifiedAtmosphere:
    r"""
    Container for the height-resolved physical state fed to :class:`NLTEStratifiedAtmosphere`.

    This is the user-input / conversion layer: every quantity may be given as a scalar, an
    array on the height grid, or a callable ``f(z_cm)``, and is sampled onto the supplied
    height grid here.  Nothing in :class:`AtmosphereParameters` is modified -- per depth a
    standard ``model.AtmosphereParameters`` is constructed on demand.

    :param model:  configured :class:`Model`.
    :param height_cm:  strictly increasing height grid [cm]; ``height_cm[0]`` is the lower
        boundary, ``height_cm[-1]`` the observer side.
    :param temperature_K:  :math:`T(z)` [K].
    :param number_density_cm3:  absorber number density :math:`N(z)` [cm^-3] (opacity scale).
    :param magnetic_field_gauss:  :math:`|B|(z)` [G].
    :param theta_B:  magnetic-field polar angle :math:`\theta_B(z)` [rad] (from vertical).
    :param chi_B:  magnetic-field azimuth :math:`\chi_B(z)` [rad].
    :param velocity_cm_sm1:  macroscopic speed :math:`|v|(z)` [cm/s].
    :param theta_v:  velocity polar angle :math:`\theta_v(z)` [rad] (from vertical).
    :param chi_v:  velocity azimuth :math:`\chi_v(z)` [rad].
    :param delta_v_turbulent_cm_sm1:  microturbulent velocity [cm/s].
    :param voigt_a:  Voigt :math:`a(z)`.
    :param continuum_opacity_cm_m1:  grey continuum absorption coefficient :math:`k_c(z)`
        [cm^-1], supplied directly (same units as the line opacity :math:`N\,\eta`). This is the
        standard slab-model treatment: the continuum opacity is an input, decoupled from the
        line, with an LTE (Planck) source (cf. Trujillo Bueno & Manso Sainz 1999, eqs. 5-7;
        Khan & Shulyak 2006, eqs. 3-6). If given, it overrides ``continuum_to_line_ratio``.
    :param continuum_to_line_ratio:  fallback grey continuum-to-line ratio used only when
        ``continuum_opacity_cm_m1`` is not supplied; sets :math:`k_c(z)` as this constant times
        the line-core opacity from the initial populations.
    """

    def __init__(
        self,
        model: Model,
        height_cm: Sequence[float],
        temperature_K: Profile,
        number_density_cm3: Profile,
        magnetic_field_gauss: Profile = 0.0,
        theta_B: Profile = 0.0,
        chi_B: Profile = 0.0,
        velocity_cm_sm1: Profile = 0.0,
        theta_v: Profile = 0.0,
        chi_v: Profile = 0.0,
        delta_v_turbulent_cm_sm1: Profile = 0.0,
        voigt_a: Profile = 0.0,
        continuum_opacity_cm_m1: Optional[Profile] = None,
        continuum_to_line_ratio: float = 0.0,
    ):
        z = np.asarray(height_cm, dtype=np.float64)
        assert z.ndim == 1 and len(z) >= 2, "height_cm must be a 1-D grid with at least 2 points."
        assert np.all(np.diff(z) > 0), "height_cm must be strictly increasing (z[0] = lower boundary)."
        assert continuum_to_line_ratio >= 0, "continuum_to_line_ratio must be non-negative."

        self.model = model
        self.height_cm = z
        self.temperature_K = _sample_profile(temperature_K, z, "temperature_K")
        self.number_density_cm3 = _sample_profile(number_density_cm3, z, "number_density_cm3")
        self.magnetic_field_gauss = _sample_profile(magnetic_field_gauss, z, "magnetic_field_gauss")
        self.theta_B = _sample_profile(theta_B, z, "theta_B")
        self.chi_B = _sample_profile(chi_B, z, "chi_B")
        self.velocity_cm_sm1 = _sample_profile(velocity_cm_sm1, z, "velocity_cm_sm1")
        self.theta_v = _sample_profile(theta_v, z, "theta_v")
        self.chi_v = _sample_profile(chi_v, z, "chi_v")
        self.delta_v_turbulent_cm_sm1 = _sample_profile(delta_v_turbulent_cm_sm1, z, "delta_v_turbulent_cm_sm1")
        self.voigt_a = _sample_profile(voigt_a, z, "voigt_a")
        self.continuum_to_line_ratio = float(continuum_to_line_ratio)
        if continuum_opacity_cm_m1 is None:
            self.continuum_opacity_cm_m1: Optional[np.ndarray] = None
        else:
            self.continuum_opacity_cm_m1 = _sample_profile(continuum_opacity_cm_m1, z, "continuum_opacity_cm_m1")
            assert np.all(self.continuum_opacity_cm_m1 >= 0), "continuum_opacity_cm_m1 must be non-negative everywhere."

        assert np.all(self.temperature_K > 0), "temperature_K must be positive everywhere."
        assert np.all(self.number_density_cm3 >= 0), "number_density_cm3 must be non-negative everywhere."

    @classmethod
    def on_uniform_grid(
        cls,
        model: Model,
        z_min_cm: float,
        z_max_cm: float,
        n_depth: int,
        **profiles: Profile,
    ) -> "StratifiedAtmosphere":
        r"""
        Convenience builder on a uniform height grid of ``n_depth`` points.
        """
        assert n_depth >= 2, "n_depth must be >= 2."
        z = np.linspace(float(z_min_cm), float(z_max_cm), n_depth)
        return cls(model=model, height_cm=z, **profiles)

    @property
    def n_depth(self) -> int:
        return len(self.height_cm)

    def atmosphere_parameters(self, i: int, macroscopic_velocity_cm_sm1: float):
        r"""
        Build the RTE/SEE ``AtmosphereParameters`` at depth ``i`` for a given LOS velocity.
        """
        return self.model.AtmosphereParameters(
            model_config=self.model.config,
            magnetic_field_gauss=self.magnetic_field_gauss[i],
            temperature_K=self.temperature_K[i],
            delta_v_turbulent_cm_sm1=self.delta_v_turbulent_cm_sm1[i],
            macroscopic_velocity_cm_sm1=macroscopic_velocity_cm_sm1,
            voigt_a=self.voigt_a[i],
        )

    def magnetic_frame_angles(self, i: int) -> Angles:
        r"""
        Angles carrying only the depth-``i`` magnetic-field orientation (for J rotation).
        """
        return Angles(chi_B=self.chi_B[i], theta_B=self.theta_B[i])

    def velocity_vector(self, i: int) -> np.ndarray:
        r"""
        Macroscopic velocity 3-vector at depth ``i`` in the fixed LL04 frame [cm/s].
        """
        st = np.sin(self.theta_v[i])
        return self.velocity_cm_sm1[i] * np.array(
            [st * np.cos(self.chi_v[i]), st * np.sin(self.chi_v[i]), np.cos(self.theta_v[i])],
            dtype=np.float64,
        )


class NLTEStratifiedAtmosphere:
    r"""
    Self-consistent NLTE synthesis through a height-stratified atmosphere, with every
    physical parameter varying continuously with geometric height: temperature, absorber
    number density, magnetic-field vector (magnitude and direction), microturbulence, Voigt
    :math:`a`, and the macroscopic-velocity vector.

    The radiation field is propagated along a Gauss-Legendre :math:`\mu` x uniform
    :math:`\phi` quadrature of rays; the emergent Stokes distribution is projected into a
    multipole radiation tensor :math:`J^K_Q` per depth (profile-weighted in frequency, per
    ray to account for the local velocity shift), the SEE is re-solved locally, and the loop
    repeats until :math:`\max|\Delta\rho|` drops below ``tolerance``.

    Formulation:

    * Geometric height ``z`` [cm] is the independent variable; ``z[0]`` is the lower boundary
      (photosphere, or zero radiation at the limb), ``z[-1]`` the observer side.
    * The opacity scale is the local absorber number density :math:`N(z)` [cm^-3]: the
      radiative-transfer code computes per-atom coefficients and multiplies by :math:`N`, so
      the optical depth follows from :math:`N(z)` and the geometry.
    * Transfer is solved in the observer (Eulerian) frame on the fixed ``z`` grid by the DELO
      method. A vertical velocity gradient enters through the Doppler-shifted absorption
      profile evaluated at the local velocity projection of each ray at each depth, valid
      while the grid resolves the gradient (the per-cell line shift stays below the line
      width; a warning is emitted otherwise).
    * The line is treated as scattering; when the atom is configured with parametrized collisions
      (:class:`ParametrizedCollisions`) the inelastic/superelastic and depolarizing rates enter the
      SEE and thermalize the line toward LTE. A grey continuum is added per depth with an LTE
      (Planck) source :math:`B(T(z))` (a constant continuum-to-line ratio, or an explicit
      :math:`k_c(z)`).

    Angles follow the LL04 convention (Fig. 5.9): a fixed reference frame with ``z`` along
    the vertical / atmosphere normal. A propagation direction is
    :math:`\hat\Omega = (\sin\theta\cos\chi, \sin\theta\sin\chi, \cos\theta)`,
    :math:`\mu = \cos\theta`; the magnetic field and macroscopic velocity are given by their
    polar angles :math:`(\theta_B,\chi_B)` and :math:`(\theta_v,\chi_v)` in the same frame.
    The line-of-sight Doppler projection feeding the profile is
    :math:`v_{\rm los} = -\hat\Omega\cdot\vec v` (see :meth:`_project`).

    References (equation-level citations are given at each formula below):

    * Lambda-iteration scheme, source functions, continuum, and the radiation-field-tensor
      moments: Trujillo Bueno & Manso Sainz (1999), ApJ, 516, 436 (DOI 10.1086/307107),
      eqs. (3)-(7) transfer + source + continuum, (10)-(11) for :math:`J^0_0, J^2_0`,
      (12)-(13) for the Lambda-iteration update (referred to below as TM99).
    * Formal solution of the polarized transfer equation (evolution operator, constant-K step):
      Landi Degl'Innocenti & Landi Degl'Innocenti (1985), Solar Phys. 97, 239, eqs. (1), (5)
      (referred to below as LandiLandi1985); see also LL04 Ch. 8.
    * Polarized transfer with line + continuum and Lambda-iteration: Khan & Shulyak (2006),
      A&A, 448, 1153, eqs. (2)-(7).
    * Polarization tensor :math:`T^K_Q`, radiation-field tensor :math:`J^K_Q`, flat-spectrum
      and moving-atom (velocity) treatment, and the transfer coefficients / number-density
      scaling: Landi Degl'Innocenti & Landolfi (2004), "Polarization in Spectral Lines" (LL04):
      Sec. 5.11 (eqs. 5.132-5.135, Table 5.2, and 5.157), Sec. 13.1 (flat spectrum),
      Secs. 12.4 / 13.2 (velocity), Ch. 7 (transfer coefficients and their :math:`N` dependence).

    :param model:  configured :class:`Model`.
    :param stratification:  :class:`StratifiedAtmosphere` height-resolved state.
    :param los_theta:  observer line-of-sight polar angle [rad] (from vertical). A tangential view
        (:math:`\theta \to 90^\circ`, :math:`\mu \to 0`) is allowed: the emergent Stokes is then the
        surface source function (Eddington-Barbier limit), computed without integrating the ray.
    :param los_chi:  observer line-of-sight azimuth [rad].
    :param los_gamma:  observer polarization reference angle [rad] (orientation of +Q).
    :param n_mu_quadrature:  total number of :math:`\mu` points; a double-Gauss rule uses
        ``n_mu_quadrature // 2`` Gauss-Legendre points on each hemisphere ([-1, 0] and [0, 1]).
        Must be even (so the two hemispheres get equal orders and no node lands on the tangential
        :math:`\mu = 0`). Comparable to TM99's ``n_mu``, which likewise counts points per hemisphere.
    :param n_phi_quadrature:  number of uniform azimuthal samples; must be >= 3 so the K = 2
        radiation-field-tensor components (azimuthal orders :math:`|Q|` up to 2) integrate correctly.
    :param max_iterations:  maximum number of iterations.
    :param tolerance:  convergence threshold on :math:`\max|\Delta\rho|`.
    :param top_incident_stokes:  Stokes incident from the observer-side boundary (defaults
        to zero).  The lower boundary uses the ``initial_stokes`` passed to :meth:`forward`.
    :param ng_acceleration:  enable Ng (1974) convergence acceleration of the Lambda-iteration:
        every ``ng_period`` iterations the last four density-matrix iterates are extrapolated to
        their fixed point by a small least-squares problem, cutting the iteration count. It operates
        on rho as an opaque vector (atom-model independent), preserves the trace normalization (the
        extrapolation weights sum to one), and leaves the converged solution unchanged.
    :param ng_period:  number of iterations between Ng extrapolations (used only when
        ``ng_acceleration`` is enabled). Larger values let the iterates settle into the asymptotic
        regime before extrapolating, reducing overshoot.
    :param ng_damping:  under-relaxation of the Ng step in (0, 1]: the accepted update is
        :math:`\rho_0 + \mathrm{ng\_damping}\,(\rho_{\rm Ng} - \rho_0)`. ``1.0`` is the full Ng step;
        lower it (e.g. ``0.5``) if the residual jumps up on the extrapolation iterations (overshoot).
    :param transfer_scheme:  ``"delo_constant"`` (first order, piecewise-constant source per cell)
        or ``"delo_linear"`` (second order, source linear across each cell).
    :param estimate_true_error:  when ``True``, converge on the estimated distance to the fixed point
        (residual divided by :math:`1-\lambda`, with :math:`\lambda` the measured contraction rate)
        rather than on the raw residual; grid-robust. With Ng it needs ``ng_period`` large enough for
        a clean measurement window per period.
    """

    def __init__(
        self,
        model: Model,
        stratification: StratifiedAtmosphere,
        los_theta: float,
        los_chi: float = 0.0,
        los_gamma: float = 0.0,
        n_mu_quadrature: int = 4,
        n_phi_quadrature: int = 4,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        top_incident_stokes: Optional[Stokes] = None,
        ng_acceleration: bool = False,
        ng_period: int = 4,
        ng_damping: float = 1.0,
        transfer_scheme: str = "delo_constant",
        estimate_true_error: bool = False,
    ):
        # A tangential observer (mu -> 0) is allowed: it is handled by the surface-source-function
        # (Eddington-Barbier) branch in forward(), so no |mu| lower bound is required here.
        assert n_mu_quadrature >= 1, "Need at least one mu quadrature point."
        assert n_mu_quadrature % 2 == 0, (
            "n_mu_quadrature must be even: the double-Gauss rule splits it evenly between the two "
            "hemispheres ([-1, 0] and [0, 1]), and no node then lands on the tangential mu = 0."
        )
        assert n_phi_quadrature >= 3, (
            "n_phi_quadrature must be >= 3: the radiation-field tensor has K = 2 components with e^{iQ phi} "
            "azimuthal dependence (|Q| up to 2), and N uniform phi points integrate e^{iQ phi} to zero only "
            "for |Q| < N. With fewer than three points the Q != 0 terms alias to a spurious J^2_{Q!=0} that "
            "injects energy into the radiation field."
        )
        assert ng_period >= 1, "ng_period must be >= 1."
        assert 0.0 < ng_damping <= 1.0, "ng_damping must be in (0, 1]."
        assert transfer_scheme in (
            "delo_constant",
            "delo_linear",
        ), "transfer_scheme must be 'delo_constant' or 'delo_linear'."
        if estimate_true_error and ng_acceleration:
            min_period = _NG_RELAX_ITERATIONS + 1 + _MIN_MEASURE_ITERATIONS
            assert ng_period >= min_period, (
                f"estimate_true_error with ng_acceleration needs ng_period >= {min_period} "
                f"({_NG_RELAX_ITERATIONS} relaxation + 1 jump + {_MIN_MEASURE_ITERATIONS} measurement "
                f"iterations per period); got ng_period = {ng_period}."
            )

        self.model = model
        self.stratification = stratification
        self.los_theta = los_theta
        self.los_chi = los_chi
        self.los_gamma = los_gamma
        self.n_mu_quadrature = n_mu_quadrature
        self.n_phi_quadrature = n_phi_quadrature
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.top_incident_stokes = top_incident_stokes
        self.ng_acceleration = ng_acceleration
        self.ng_period = ng_period
        self.ng_damping = ng_damping
        self.transfer_scheme = transfer_scheme
        self.estimate_true_error = estimate_true_error

        # Diagnostics populated by forward()
        self.rho_grid: Optional[List[BaseRho]] = None
        self.radiation_tensor_grid: Optional[List[BaseRadiationTensor]] = None
        self.tau_grid: Optional[np.ndarray] = None
        self.iterations_used: Optional[int] = None
        self.final_residual: Optional[float] = None
        self.residual_history: List[float] = []
        self.final_true_error: Optional[float] = None
        self.lambda_estimate: Optional[float] = None

        # Transition ids and rest frequencies for the per-transition profile weights (set in
        # forward()); the profiles themselves are built per (ray, depth).
        self._recon_transition_ids: List[str] = []
        self._recon_centers: Optional[np.ndarray] = None

    @property
    def model_signature(self) -> str:
        r"""
        Identifier of the atom model, stored in the :class:`NLTEState` and checked on warm start.
        """
        config_type = type(self.model.config)
        return f"{config_type.__module__}.{config_type.__qualname__}"

    def get_state(self) -> NLTEState:
        r"""
        The current density-matrix grid as a reusable :class:`NLTEState` (available after
        :meth:`forward`).
        """
        assert self.rho_grid is not None, "forward() must run before get_state()."
        return NLTEState.from_rho_grid(
            height_cm=self.stratification.height_cm, rho_grid=self.rho_grid, model_signature=self.model_signature
        )

    @log_method
    def forward(
        self,
        initial_stokes: Stokes,
        initial_state: Optional[NLTEState] = None,
        on_iteration: Optional[Callable[[int, Stokes], None]] = None,
    ) -> Stokes:
        r"""
        Run the NLTE loop and return the emergent Stokes along the observer ray.

        ``initial_state`` warm-starts from a previous :class:`NLTEState` instead of the LTE guess;
        ``on_iteration(iteration, emergent)`` is called after each iteration with the current
        emergent Stokes (e.g. to record a convergence history). Retrieve the final state with
        :meth:`get_state`.
        """
        nu = initial_stokes.nu
        strat = self.stratification
        n_z = strat.n_depth
        z = strat.height_cm
        mu_obs = float(np.cos(self.los_theta))

        self.iterations_used = None
        self.final_residual = None
        self.residual_history = []
        self.final_true_error = None
        self.lambda_estimate = None

        see: BaseSEE = self.model.StatisticalEquilibriumEquations.from_model_config(self.model.config)
        rte: BaseRTE = self.model.RadiativeTransferEquations.from_model_config(self.model.config, nu=nu)
        # Opacity is carried by the per-depth number density: the line transfer coefficients are
        # proportional to the lower-level number density N (LL04 Ch. 7), so set rte.N = N(z_i)
        # per call. N is the final factor applied after the cached operator, so the operator cache
        # stays valid.
        rte.N = 1.0
        if hasattr(rte, "use_operator_cache"):
            rte.use_operator_cache = True

        # Per-depth precomputed quantities.
        N = strat.number_density_cm3
        bp_per_z = [get_planck_BP(nu_sm1=nu, temperature_K=strat.temperature_K[i]) for i in range(n_z)]
        b_angles = [strat.magnetic_frame_angles(i) for i in range(n_z)]
        v_vectors = [strat.velocity_vector(i) for i in range(n_z)]
        # SEE rates are built from the frequency-integrated J, so the macroscopic velocity is
        # irrelevant there; use zero.
        see_params = [strat.atmosphere_parameters(i, 0.0) for i in range(n_z)]

        # Observer geometry (LOS shared by all depths; B orientation per depth).
        omega_obs = self._ray_direction(self.los_theta, self.los_chi)
        obs_angles = [
            Angles(
                chi=self.los_chi, theta=self.los_theta, gamma=self.los_gamma,
                chi_B=strat.chi_B[i], theta_B=strat.theta_B[i],
            )
            for i in range(n_z)
        ]  # fmt: skip

        # 1. Initial guess: solve the SEE per depth with an isotropic Planck radiation tensor.
        rho_grid: List[BaseRho] = []
        for i in range(n_z):
            rad_planck = self.model.RadiationTensor.from_model_config(self.model.config).fill_planck(
                temperature_K=strat.temperature_K[i]
            )
            see.fill_all_equations(
                atmosphere_parameters=see_params[i],
                radiation_tensor_in_magnetic_frame=rad_planck.rotate_to_magnetic_frame(angles=b_angles[i]),
            )
            rho_grid.append(see.get_solution())

        # 1b. Warm start: overwrite the LTE guess with a previous solution (resampled to this grid).
        # The isotropic-Planck solve above still runs so rho_grid carries the correct per-depth
        # coherence structure; only the values are replaced.
        if initial_state is not None:
            initial_state.check_compatible(self.model_signature, coherence_keys=list(rho_grid[0].data))
            initial_state.interpolate_to(z).apply_to_templates(rho_grid)

        # 2. Per-depth line-core opacity along the observer ray (absolute, includes N), used for
        #    the optical-depth diagnostic and, when no explicit k_c(z) is supplied, to scale the
        #    grey continuum from the continuum-to-line ratio.
        eta_peak = np.zeros(n_z)
        for i in range(n_z):
            rte.N = float(N[i])
            v_los = self._project(v_vectors[i], omega_obs)
            rtc = rte.calculate_all_coefficients(
                atmosphere_parameters=strat.atmosphere_parameters(i, v_los),
                angles=obs_angles[i],
                rho=rho_grid[i],
            )
            eta_peak[i] = float(np.max(np.abs(np.real(rtc.get_eta_I()))))
        assert float(np.max(eta_peak)) > 0, (
            "Line opacity along the observer ray is zero for the initial guess. "
            "Check that the frequency grid covers the transition and N(z) > 0."
        )
        # Grey continuum opacity k_c(z): use the user-supplied absolute profile if given
        # (standard slab-model input; TM99 eqs. 5-7, Khan & Shulyak 2006 eqs. 3-6), otherwise
        # fall back to the continuum-to-line ratio times the line-core opacity.
        if strat.continuum_opacity_cm_m1 is not None:
            k_c_per_z = np.asarray(strat.continuum_opacity_cm_m1, dtype=np.float64)
        else:
            k_c_per_z = strat.continuum_to_line_ratio * eta_peak  # [n_z]

        # Vertical line optical depth (diagnostic, observer-independent): tau = int eta dz. The
        # observer-ray depth is this divided by |mu_obs|; storing the vertical value keeps the
        # diagnostic finite for a tangential (mu = 0) line of sight.
        d_tau = 0.5 * (eta_peak[1:] + eta_peak[:-1]) * np.diff(z)
        self.tau_grid = np.concatenate([[0.0], np.cumsum(d_tau)])

        # Transition rest frequencies, used to build the per-transition absorption profiles that
        # weight the J^K_Q reconstruction (LL04 Sec. 13.1). Depends only on the atom, so once.
        recon_transitions = list(
            self.model.RadiationTensor.from_model_config(self.model.config).transition_registry.transitions.values()
        )
        self._recon_transition_ids = [t.transition_id for t in recon_transitions]
        self._recon_centers = np.array(
            [t.get_mean_transition_frequency_sm1() for t in recon_transitions], dtype=np.float64
        )

        # 3. Quadrature rays, with the per-ray T^K_Q* and per-(ray, depth) projected velocity.
        rays = self._build_quadrature_rays()
        thermal_v_per_z = np.array([p.delta_v_thermal_cm_sm1 for p in see_params], dtype=np.float64)
        self._warn_if_velocity_underresolved(rays=rays, v_vectors=v_vectors, thermal_v_per_z=thermal_v_per_z, nu=nu)
        t_conj_per_ray = [self._t_conj_for_ray(ray) for ray in rays]
        ray_params = []
        ray_angles = []
        for ray in rays:
            omega = self._ray_direction(ray["theta"], ray["chi"])
            ray_params.append([strat.atmosphere_parameters(i, self._project(v_vectors[i], omega)) for i in range(n_z)])
            ray_angles.append(
                [
                    Angles(
                        chi=ray["chi"], theta=ray["theta"], gamma=0.0, chi_B=strat.chi_B[i], theta_B=strat.theta_B[i]
                    )
                    for i in range(n_z)
                ]
            )

        # Per-(ray, depth) normalized absorption profile of each transition (LL04 eq. 5.44),
        # used as the frequency weight in the J^K_Q reconstruction. These depend only on the
        # geometry/atmosphere (not on rho), so precompute them once for the whole iteration.
        profile_weights_per_ray = [
            [self._profile_weights_at(nu, ray_params[r][i]) for i in range(n_z)] for r in range(len(rays))
        ]

        bottom_bc = initial_stokes
        top_bc = self.top_incident_stokes if self.top_incident_stokes is not None else Stokes.from_zeros(nu_sm1=nu)

        # Emergent Stokes along the observer ray for a given rho grid (independent of rho apart from
        # that grid). A tangential view (|mu| -> 0) is the Eddington-Barbier limit I(0) = S(tau=0);
        # otherwise the ray is integrated. Reused for the per-iteration callback and the final value.
        obs_params = [strat.atmosphere_parameters(i, self._project(v_vectors[i], omega_obs)) for i in range(n_z)]

        def emergent_stokes_for(current_rho_grid: List[BaseRho]) -> Stokes:
            if abs(mu_obs) < _MU_TANGENTIAL_THRESHOLD:
                e = self._tangential_emergent(
                    i_surface=n_z - 1,
                    rte=rte,
                    rho_grid=current_rho_grid,
                    params_per_z=obs_params,
                    angles_per_z=obs_angles,
                    number_density=N,
                    k_c_per_z=k_c_per_z,
                    bp_per_z=bp_per_z,
                )
            else:
                stokes_z = self._propagate_ray(
                    rho_grid=current_rho_grid,
                    z=z,
                    mu_n=mu_obs,
                    rte=rte,
                    params_per_z=obs_params,
                    angles_per_z=obs_angles,
                    number_density=N,
                    k_c_per_z=k_c_per_z,
                    bp_per_z=bp_per_z,
                    bottom_bc=bottom_bc,
                    top_bc=top_bc,
                )
                e = stokes_z[-1] if mu_obs > 0 else stokes_z[0]
            return Stokes(nu=nu, I=real(e[0]), Q=real(e[1]), U=real(e[2]), V=real(e[3]))

        # 4. Lambda-iteration: rho(old) -> formal solution for I along each ray -> reconstruct
        # J^K_Q -> re-solve the SEE for rho(new). Convergence is tested on the maximum coherence
        # change max|delta rho| (TM99 eqs. 12-13 for the update; their R_c, Sec. 3.1).
        rho_history: List[List[BaseRho]] = []  # last few iterates, for optional Ng acceleration
        measure_residuals: List[float] = []  # clean-decay residuals, for optional true-error estimation
        converged = False
        for iteration in range(self.max_iterations):
            stokes_per_ray: List[np.ndarray] = []
            for r in range(len(rays)):
                stokes_z = self._propagate_ray(
                    rho_grid=rho_grid, z=z, mu_n=rays[r]["mu"], rte=rte,
                    params_per_z=ray_params[r], angles_per_z=ray_angles[r],
                    number_density=N, k_c_per_z=k_c_per_z, bp_per_z=bp_per_z,
                    bottom_bc=bottom_bc, top_bc=top_bc,
                )  # fmt: skip
                stokes_per_ray.append(stokes_z)

            new_rho_grid: List[BaseRho] = []
            radiation_tensors: List[BaseRadiationTensor] = []
            for i in range(n_z):
                radiation_tensor_i = self._reconstruct_radiation_tensor(
                    rays=rays, stokes_per_ray=stokes_per_ray,
                    profile_weights_per_ray=profile_weights_per_ray,
                    i_z=i, t_conj_per_ray=t_conj_per_ray,
                )  # fmt: skip
                radiation_tensors.append(radiation_tensor_i)
                see.fill_all_equations(
                    atmosphere_parameters=see_params[i],
                    radiation_tensor_in_magnetic_frame=radiation_tensor_i.rotate_to_magnetic_frame(angles=b_angles[i]),
                )
                new_rho_grid.append(see.get_solution())
            self.radiation_tensor_grid = radiation_tensors  # diagnostic: last iteration's J^K_Q per depth

            if self.ng_acceleration:
                rho_history.append(new_rho_grid)
                if len(rho_history) > 4:
                    rho_history.pop(0)
                if len(rho_history) == 4 and (iteration + 1) % self.ng_period == 0:
                    accelerated = self._ng_accelerate(rho_history, self.ng_damping)
                    if accelerated is not None:
                        new_rho_grid = accelerated
                        rho_history[-1] = accelerated

            residual = self._rho_grid_diff(rho_grid, new_rho_grid)
            rho_grid = new_rho_grid
            self.iterations_used = iteration + 1
            self.final_residual = residual
            self.residual_history.append(residual)
            if on_iteration is not None:
                on_iteration(iteration, emergent_stokes_for(rho_grid))

            if not self.estimate_true_error:
                logging.info(f"NLTE (stratified) iteration {iteration}: residual = {residual:.3e}")
                if residual < self.tolerance:
                    converged = True
                    logging.info(
                        "NLTE (stratified) converged after %d iterations " "(final residual=%.3e, tolerance=%.3e)",
                        self.iterations_used,
                        residual,
                        self.tolerance,
                    )
                    break
                continue

            # Stop on the estimated distance to the fixed point rather than on the step size. Only the
            # clean-decay iterations feed the rate estimate: plain iterations always, or the measure
            # window of each Ng period (past the jump and its relaxation).
            in_measure_window = (
                not self.ng_acceleration or _NG_RELAX_ITERATIONS <= iteration % self.ng_period <= self.ng_period - 2
            )
            measure_residuals = (measure_residuals + [residual])[-_LAMBDA_WINDOW:] if in_measure_window else []
            true_error, lambda_hat = self._estimate_true_error(measure_residuals)
            self.final_true_error = true_error
            self.lambda_estimate = lambda_hat
            if true_error is None:
                logging.info(f"NLTE (stratified) iteration {iteration}: residual = {residual:.3e}")
            else:
                logging.info(
                    "NLTE (stratified) iteration %d: residual = %.3e, estimated error = %.3e (lambda = %.3f)",
                    iteration,
                    residual,
                    true_error,
                    lambda_hat,
                )
                if true_error < self.tolerance:
                    converged = True
                    logging.info(
                        "NLTE (stratified) converged after %d iterations "
                        "(final residual=%.3e, estimated error=%.3e, tolerance=%.3e)",
                        self.iterations_used,
                        residual,
                        true_error,
                        self.tolerance,
                    )
                    break

        if not converged:
            logging.warning(
                "NLTE (stratified) stopped after max_iterations=%d before meeting tolerance=%.3e "
                "(final residual=%s, estimated error=%s)",
                self.max_iterations,
                self.tolerance,
                "None" if self.final_residual is None else f"{self.final_residual:.3e}",
                "None" if self.final_true_error is None else f"{self.final_true_error:.3e}",
            )

        self.rho_grid = rho_grid
        emergent = emergent_stokes_for(rho_grid)

        if hasattr(rte, "clear_operator_cache"):
            rte.clear_operator_cache()

        return emergent

    # ------------------------------------------------------------------ geometry / velocity

    @staticmethod
    def _ray_direction(theta: float, chi: float) -> np.ndarray:
        r"""
        Propagation unit vector :math:`\hat\Omega` in the fixed LL04 frame (z = vertical).
        """
        st = np.sin(theta)
        return np.array([st * np.cos(chi), st * np.sin(chi), np.cos(theta)], dtype=np.float64)

    @staticmethod
    def _project(v_vector: np.ndarray, omega: np.ndarray) -> float:
        r"""
        Line-of-sight velocity projection feeding the absorption profile,
        :math:`v_{\rm los} = -\,\hat\Omega\cdot\vec v`, with :math:`\hat\Omega` the photon
        propagation (toward-observer) direction.

        The minus sign matches the profile convention in the RTE (``_phi``:
        :math:`\nu = \nu_i (1 - v_{\rm los}/c)`, i.e. positive :math:`v_{\rm los}` = redshift):
        plasma moving toward the observer (a velocity component along :math:`\hat\Omega`) gives
        :math:`v_{\rm los} < 0` and therefore a blueshift. Equivalently, :math:`v_{\rm los}` is
        the projection onto the into-the-medium direction (receding-positive).

        Physically this is the Doppler shift of the absorption profile of an atom moving with the
        local macroscopic velocity, evaluated per ray (the moving-atom picture of LL04 Sec. 13.1;
        the resulting Doppler shift of the radiation-field tensor is discussed in LL04 Sec. 12.4).
        A single deterministic bulk velocity per depth is used; the full velocity-space
        redistribution of LL04 Sec. 13.2 is not implemented.
        """
        return -float(np.dot(v_vector, omega))

    def _build_quadrature_rays(self) -> List[Dict]:
        r"""
        Double-Gauss :math:`\mu` x uniform :math:`\phi` quadrature.  Each ray weight already
        includes the :math:`1/(4\pi)` normalization of :math:`J^K_Q`.

        The :math:`\mu` integral uses an independent Gauss-Legendre rule on each hemisphere
        ([-1, 0] and [0, 1]) rather than a single rule over [-1, 1].  At the surface the radiation
        field has a kink at :math:`\mu = 0` (up-going rays see the slab, down-going rays see the
        boundary), which destroys the spectral convergence of a single rule over [-1, 1] and biases
        the surface anisotropy (hence :math:`\rho^2_0/\rho^0_0`) low.  Splitting at :math:`\mu = 0`
        puts the kink on a subinterval boundary, so each half is smooth and converges spectrally --
        this is the standard slab-RT choice (and TM99's, whose ``n_mu`` counts points per hemisphere).
        """
        # n_mu_quadrature total points, split evenly between the two hemispheres.
        nodes_11, weights_11 = np.polynomial.legendre.leggauss(self.n_mu_quadrature // 2)
        mus = []
        mu_weights = []
        for lower, upper in ((-1.0, 0.0), (0.0, 1.0)):
            half_width = 0.5 * (upper - lower)
            midpoint = 0.5 * (upper + lower)
            mus.extend(half_width * nodes_11 + midpoint)
            mu_weights.extend(half_width * weights_11)
        mus = np.array(mus)
        mu_weights = np.array(mu_weights)
        phi_grid = np.linspace(0.0, 2 * np.pi, self.n_phi_quadrature, endpoint=False)
        phi_weight_each = 2 * np.pi / self.n_phi_quadrature

        rays: List[Dict] = []
        for mu, w_mu in zip(mus, mu_weights):
            for phi in phi_grid:
                rays.append(
                    {
                        "theta": float(np.arccos(mu)),
                        "chi": float(phi),
                        "mu": float(mu),
                        "weight": float(w_mu) * phi_weight_each / (4 * np.pi),
                    }
                )
        return rays

    def _t_conj_for_ray(self, ray: Dict) -> Dict:
        r"""
        Pre-compute :math:`T^{K*}_Q(i, \Omega)` for one ray (gamma = 0 for quadrature rays).
        """
        kq_pairs = list(nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")))
        return {
            (int(K), int(Q)): np.array(
                [
                    T_K_Q(
                        K=int(K),
                        Q=int(Q),
                        stokes_component_index=k,
                        chi=ray["chi"],
                        theta=ray["theta"],
                        gamma=0.0,
                    ).conjugate()
                    for k in range(4)
                ],
                dtype=np.complex128,
            )  # fmt: skip
            for (K, Q) in kq_pairs
        }

    def _warn_if_velocity_underresolved(
        self, rays: List[Dict], v_vectors: List[np.ndarray], thermal_v_per_z: np.ndarray, nu: np.ndarray
    ) -> None:
        r"""
        Warn if the observer-frame DELO grid under-resolves the velocity field: either the
        per-cell velocity shift exceeds the local thermal width (gradient unresolved) or the
        largest projection shifts the line off the frequency grid.
        """
        n_z = len(thermal_v_per_z)
        max_abs_v_los = 0.0
        gradient_warned = False
        # Compare against the smaller thermal width across each cell (the more stringent bound).
        cell_width = np.minimum(thermal_v_per_z[1:], thermal_v_per_z[:-1])
        for ray in rays:
            omega = self._ray_direction(ray["theta"], ray["chi"])
            v_los = np.array([self._project(v_vectors[i], omega) for i in range(n_z)])
            max_abs_v_los = max(max_abs_v_los, float(np.max(np.abs(v_los))))
            if not gradient_warned and np.any(np.abs(np.diff(v_los)) > cell_width):
                logging.warning(
                    "Macroscopic-velocity shift across a height cell exceeds the local thermal line "
                    "width along a quadrature ray; refine the height grid to resolve the velocity gradient."
                )
                gradient_warned = True

        max_shift_nu = float(np.mean(nu)) * max_abs_v_los / c_cm_sm1
        if max_shift_nu > 0.5 * (float(np.max(nu)) - float(np.min(nu))):
            logging.warning(
                "Largest macroscopic-velocity Doppler shift is comparable to the frequency-grid span; "
                "the shifted line may fall outside the grid. Widen the nu range."
            )

    # ------------------------------------------------------------------ transfer

    def _propagate_ray(
        self,
        rho_grid: List[BaseRho],
        z: np.ndarray,
        mu_n: float,
        rte: BaseRTE,
        params_per_z: List,
        angles_per_z: List[Angles],
        number_density: np.ndarray,
        k_c_per_z: np.ndarray,
        bp_per_z: List[np.ndarray],
        bottom_bc: Stokes,
        top_bc: Stokes,
    ):
        r"""
        DELO-propagate Stokes through the ``z`` grid along one ray, returning
        ``stokes[n_z, 4, n_nu]``.  ``z[0]`` is the lower boundary, ``z[-1]`` the observer side.
        ``mu_n > 0`` propagates upward (boundary at ``z[0]``); ``mu_n < 0`` downward (boundary at
        ``z[-1]``).
        """
        n_z = len(z)
        nu = bottom_bc.nu
        n_nu = len(nu)
        assert abs(mu_n) >= 1e-6, _ERR_TANGENTIAL_MU

        # Coefficients at every depth (line, absolute via rte.N, plus grey continuum).
        K_per_z = []
        eps_per_z = []
        for i in range(n_z):
            rte.N = float(number_density[i])
            rtc = rte.calculate_all_coefficients(
                atmosphere_parameters=params_per_z[i], angles=angles_per_z[i], rho=rho_grid[i]
            )
            K = rtc.K_z()  # [n_nu, 4, 4]
            eps = rtc.epsilon_z()[:, :, 0]  # [n_nu, 4]
            # Grey continuum: unpolarized continuum opacity on the diagonal of K and an LTE
            # (Planck) continuum source added to Stokes I. Cf. Trujillo Bueno & Manso Sainz
            # (1999) eqs. (5)-(7) with S_c = Planck, and Khan & Shulyak (2006, A&A 448, 1153)
            # eqs. (3)-(6): K = (kappa_c + ...) 1 + line, epsilon_c = kappa_c B(T).
            for k in range(4):
                K[:, k, k] += k_c_per_z[i]
            eps[:, 0] += k_c_per_z[i] * bp_per_z[i]
            K_per_z.append(K)
            eps_per_z.append(eps)

        stokes = np.zeros((n_z, 4, n_nu), dtype=np.complex128)
        linear = self.transfer_scheme == "delo_linear"
        if mu_n > 0:
            stokes[0] = np.stack([bottom_bc.I, bottom_bc.Q, bottom_bc.U, bottom_bc.V])
            for i in range(0, n_z - 1):
                ds = (z[i + 1] - z[i]) / abs(mu_n)
                if linear:
                    stokes[i + 1] = self._delo_linear_step(
                        K_per_z[i], eps_per_z[i], K_per_z[i + 1], eps_per_z[i + 1], stokes[i], ds
                    )
                else:
                    stokes[i + 1] = self._delo_matrix_step(K_per_z[i], eps_per_z[i], stokes[i], ds)
        else:
            stokes[-1] = np.stack([top_bc.I, top_bc.Q, top_bc.U, top_bc.V])
            for i in range(n_z - 1, 0, -1):
                ds = (z[i] - z[i - 1]) / abs(mu_n)
                if linear:
                    stokes[i - 1] = self._delo_linear_step(
                        K_per_z[i], eps_per_z[i], K_per_z[i - 1], eps_per_z[i - 1], stokes[i], ds
                    )
                else:
                    stokes[i - 1] = self._delo_matrix_step(K_per_z[i], eps_per_z[i], stokes[i], ds)
        return stokes

    def _tangential_emergent(
        self,
        i_surface: int,
        rte: BaseRTE,
        rho_grid: List[BaseRho],
        params_per_z: List,
        angles_per_z: List[Angles],
        number_density: np.ndarray,
        k_c_per_z: np.ndarray,
        bp_per_z: List[np.ndarray],
    ) -> np.ndarray:
        r"""
        Emergent Stokes ``[4, n_nu]`` for a tangential line of sight (:math:`\mu \to 0`), the
        Eddington-Barbier limit :math:`I(0, \mu\to 0) = S(\tau = 0)`: the surface source function
        :math:`S = K^{-1}\epsilon` at the observer surface, evaluated with the tangential
        (:math:`\theta = 90^\circ`) transfer coefficients. No path is integrated, so the diverging
        tangential path length never appears.
        """
        rte.N = float(number_density[i_surface])
        rtc = rte.calculate_all_coefficients(
            atmosphere_parameters=params_per_z[i_surface], angles=angles_per_z[i_surface], rho=rho_grid[i_surface]
        )
        K = rtc.K_z()  # [n_nu, 4, 4]
        eps = rtc.epsilon_z()[:, :, 0]  # [n_nu, 4]
        for k in range(4):
            K[:, k, k] += k_c_per_z[i_surface]
        eps[:, 0] += k_c_per_z[i_surface] * bp_per_z[i_surface]
        return self._delo_source_function(K, eps).T  # [4, n_nu]

    @staticmethod
    def _delo_source_function(K: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        r"""
        Source function :math:`S = K^{-1}\epsilon` (LandiLandi1985 eq. 1), batched over frequency,
        with a per-frequency pseudo-inverse fallback on a singular ``K``.
        """
        try:
            return np.linalg.solve(K, epsilon[:, :, np.newaxis])[:, :, 0]  # [n_nu, 4]
        except np.linalg.LinAlgError:
            source = np.empty_like(epsilon)
            for n in range(K.shape[0]):
                try:
                    source[n] = np.linalg.solve(K[n], epsilon[n])
                except np.linalg.LinAlgError:
                    source[n] = np.linalg.pinv(K[n]) @ epsilon[n]
            return source

    @staticmethod
    def _delo_matrix_step(K: np.ndarray, epsilon: np.ndarray, current_stokes: np.ndarray, ds: float) -> np.ndarray:
        r"""
        One step of the polarized formal solution over physical length ``ds``, with the
        propagation matrix K and the source function S taken constant across the cell:

        .. math::

            S = K^{-1}\epsilon, \qquad \mathrm{Stokes}(s+ds) = S + e^{-K\,ds}\,(\mathrm{Stokes}(s) - S)

        batched over frequency.  ``K`` is ``[n_nu, 4, 4]``, ``epsilon`` ``[n_nu, 4]``,
        ``current_stokes`` ``[4, n_nu]``.

        This is the evolution-operator solution of the transfer equation
        :math:`\mathrm dI/\mathrm ds = -K\,(I - S)` for constant K: the evolution operator is
        :math:`O(s+ds, s) = e^{-K\,ds}` (Landi Degl'Innocenti & Landi Degl'Innocenti 1985,
        Solar Phys. 97, 239; eq. 1 for the transfer equation, eq. 5 for the constant-K operator),
        evaluated by diagonalization :math:`O = X\,\mathrm{diag}(e^{-k_i\,ds})\,X^{-1}` (ibid.,
        p. 241; LL04 Sec. 8.4). Integrating the constant-source term over the cell yields the
        :math:`S + e^{-K\,ds}(I - S)` form (DELO with constant source; LL04 Secs. 8.2, 9.15).
        """
        # Source function S = K^{-1} epsilon, i.e. the S such that dI/ds = -K(I - S)
        # (LandiLandi1985 eq. 1).
        S = NLTEStratifiedAtmosphere._delo_source_function(K, epsilon)

        # Evolution operator e^{-K ds} by diagonalization O = X diag(e^{-k_i ds}) X^{-1}
        # (LandiLandi1985 eq. 5 and p. 241; LL04 Sec. 8.4).
        lam, V = np.linalg.eig(-K * ds)  # lam [n_nu, 4], V [n_nu, 4, 4]
        expM = np.real(V @ (np.exp(lam)[:, :, np.newaxis] * np.linalg.inv(V)))  # [n_nu, 4, 4]

        current = current_stokes.T[:, :, np.newaxis]  # [n_nu, 4, 1]
        new = S[:, :, np.newaxis] + np.einsum("nij,njk->nik", expM, current - S[:, :, np.newaxis])
        return new[:, :, 0].T  # [4, n_nu]

    @staticmethod
    def _delo_linear_step(
        K_up: np.ndarray,
        eps_up: np.ndarray,
        K_down: np.ndarray,
        eps_down: np.ndarray,
        current_stokes: np.ndarray,
        ds: float,
    ) -> np.ndarray:
        r"""
        One second-order DELO step over ``ds`` with the source varying linearly across the cell and
        the evolution operator using the cell-mean opacity :math:`\bar K = \tfrac12(K_{\rm up}+K_{\rm down})`:

        .. math::

            \mathrm{Stokes}_{\rm out} = E\,\mathrm{Stokes}_{\rm in} + (P - E)\,S_{\rm up} + (I - P)\,S_{\rm down},
            \quad E = e^{-\bar K\,ds},\; P = (\bar K\,ds)^{-1}(I - E),

        with the local source functions :math:`S = K^{-1}\epsilon` at each node. Reduces to the
        constant-source step when the two nodes coincide (Rees, Murphy & Durrant 1989, ApJ 339, 1093).
        """
        S_up = NLTEStratifiedAtmosphere._delo_source_function(K_up, eps_up)  # [n_nu, 4]
        S_down = NLTEStratifiedAtmosphere._delo_source_function(K_down, eps_down)  # [n_nu, 4]

        K_bar = 0.5 * (K_up + K_down)
        lam, V = np.linalg.eig(-K_bar * ds)  # eigenvalues of -K_bar ds
        V_inv = np.linalg.inv(V)
        # E = e^{-K_bar ds} and P = (K_bar ds)^{-1}(I - E), each per eigenvalue: e^{lam} and
        # g(lam) = (e^{lam} - 1)/lam (-> 1 as lam -> 0; small-|lam| Taylor avoids cancellation).
        small = np.abs(lam) < 1e-6
        g = np.where(small, 1.0 + lam * (0.5 + lam / 6.0), (np.exp(lam) - 1.0) / np.where(small, 1.0, lam))
        E = np.real(V @ (np.exp(lam)[:, :, np.newaxis] * V_inv))  # [n_nu, 4, 4]
        P = np.real(V @ (g[:, :, np.newaxis] * V_inv))  # [n_nu, 4, 4]

        psi_up = P - E
        psi_down = np.eye(4)[np.newaxis] - P
        current = current_stokes.T[:, :, np.newaxis]  # [n_nu, 4, 1]
        new = E @ current + psi_up @ S_up[:, :, np.newaxis] + psi_down @ S_down[:, :, np.newaxis]
        return new[:, :, 0].T  # [4, n_nu]

    # ------------------------------------------------------------------ radiation tensor

    def _reconstruct_radiation_tensor(
        self,
        rays: List[Dict],
        stokes_per_ray: List[np.ndarray],
        profile_weights_per_ray: List[List[Dict]],
        i_z: int,
        t_conj_per_ray: List[Dict],
    ) -> BaseRadiationTensor:
        r"""
        Reconstruct the radiation-field tensor :math:`J^K_Q` per transition at depth ``i_z``:

        .. math::

            J^K_Q = \sum_n w_n \sum_i T^{K*}_Q(i, \Omega_n) \int d\nu\, \phi_{n,t}(\nu)\, S_i(\nu, \Omega_n)

        This is the frequency- and angle-average of the Stokes vector weighted by the
        polarization tensor :math:`T^K_Q(i, \Omega)` (LL04 Sec. 5.11, eqs. 5.132-5.135 and
        Table 5.2; radiation-field tensor LL04 eq. 5.157). The angular integral
        :math:`\oint \mathrm d\Omega/4\pi` is discretized by the Gauss-Legendre :math:`\mu` x
        uniform :math:`\phi` quadrature, with ``ray["weight"]`` carrying :math:`w_n`. The
        frequency weight :math:`\phi_{n,t}` is each transition's *own* normalized absorption
        profile at this ray and depth (flat-spectrum approximation, complete frequency
        redistribution; LL04 Sec. 13.1), including its local velocity shift (moving-atom Doppler
        effect on the radiation tensor; LL04 Secs. 12.4, 13.2). Overlapping profiles are summed,
        not partitioned. For the axisymmetric K=0,2 / Q=0 case this reduces to Trujillo Bueno &
        Manso Sainz (1999) eqs. (10)-(11).
        """
        rad_tens = self.model.RadiationTensor.from_model_config(self.model.config)
        kq_pairs = list(nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")))

        accumulator: Dict = {}
        for r, ray in enumerate(rays):
            weights = profile_weights_per_ray[r][i_z]
            t_conj = t_conj_per_ray[r]
            w_ray = ray["weight"]
            for transition in rad_tens.transition_registry.transitions.values():
                s_eff = stokes_per_ray[r][i_z, :, :] @ weights[transition.transition_id]  # [4]
                for K, Q in kq_pairs:
                    key = (transition.transition_id, int(K), int(Q))
                    contribution = w_ray * complex(np.sum(t_conj[(int(K), int(Q))] * s_eff))
                    accumulator[key] = accumulator.get(key, 0.0 + 0.0j) + contribution

        for (transition_id, K_int, Q_int), value in accumulator.items():
            # J^K_0 is real by symmetry; drop accumulated numerical imaginary noise.
            stored = complex(value.real, 0.0) if Q_int == 0 else complex(value)
            rad_tens.data[rad_tens.get_key(transition_id=transition_id, K=K_int, Q=Q_int)] = stored
        rad_tens._df = None
        return rad_tens

    def _profile_weights_at(self, nu: np.ndarray, params) -> Dict:
        r"""
        Normalized absorption profile of each transition at one (ray, depth), the frequency
        weight :math:`\hat\phi_t(\nu)` (sums to 1) used in the :math:`J^K_Q` reconstruction.

        Each transition contributes its *own* Voigt profile; overlapping profiles coexist (no
        partition of the frequency axis). This is the flat-spectrum treatment of well-separated
        or blended lines (LL04 Sec. 13.1); it is exact for a single isolated line.
        """
        return {
            tid: self._absorption_profile(
                nu=nu,
                nu0=self._recon_centers[k],
                delta_v_thermal_cm_sm1=params.delta_v_thermal_cm_sm1,
                macroscopic_velocity_cm_sm1=params.macroscopic_velocity_cm_sm1,
                voigt_a=params.voigt_a,
            )
            for k, tid in enumerate(self._recon_transition_ids)
        }

    @staticmethod
    def _absorption_profile(
        nu: np.ndarray,
        nu0: float,
        delta_v_thermal_cm_sm1: float,
        macroscopic_velocity_cm_sm1: float,
        voigt_a: float,
    ) -> np.ndarray:
        r"""
        Normalized Voigt absorption profile of a single transition (discretized to sum to 1).

        Follows LL04 eq. (5.44): :math:`\phi = H(v - v_A, a) / (\sqrt\pi\,\Delta\nu_D)`, with the
        reduced variables of eqs. (5.42)-(5.43): :math:`\Delta\nu_D = \nu_0\,w_T/c` the Doppler
        width, :math:`v = (\nu_0 - \nu)/\Delta\nu_D` the reduced frequency,
        :math:`v_A = w_A/w_T` the bulk-velocity shift, and :math:`a` the damping constant; ``H``
        is the Voigt function (LL04 eq. 5.45). This matches the RTE profile ``_phi`` used in the
        transfer opacity. The LL04 sign convention (eq. 5.41: :math:`w_A > 0` for a receding
        flow, hence a redshift) is consistent with :meth:`_project`. The magnetic (Zeeman) shift
        :math:`v_B` of eq. (5.44) is omitted: the flat-spectrum pumping term uses the unshifted
        Doppler profile, valid while the Zeeman splitting is small compared to the line width.
        """
        delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
        v = (nu0 - nu) / delta_nu_D  # reduced frequency
        v_A = macroscopic_velocity_cm_sm1 / delta_v_thermal_cm_sm1  # bulk-velocity shift
        phi = np.maximum(np.real(voigt(nu=v - v_A, a=voigt_a)), 0.0)  # H(v - v_A, a)
        total = float(phi.sum())
        if total <= 0.0:
            phi = np.zeros_like(nu)
            phi[int(np.argmin(np.abs(nu - nu0)))] = 1.0
            return phi
        return phi / total

    @staticmethod
    def _rho_grid_diff(rho_grid_old: List[BaseRho], rho_grid_new: List[BaseRho]) -> float:
        r"""
        Maximum :math:`|\rho_{\rm new} - \rho_{\rm old}|` over the grid and coherences.
        """
        max_diff = 0.0
        for rho_old, rho_new in zip(rho_grid_old, rho_grid_new):
            for key, val_new in rho_new.data.items():
                d = abs(val_new - rho_old.data.get(key, 0.0))
                if d > max_diff:
                    max_diff = d
        return float(max_diff)

    @staticmethod
    def _estimate_true_error(residuals: List[float]):
        r"""
        Estimate the remaining distance to the fixed point from the geometric decay of consecutive
        residuals: with contraction rate :math:`\hat\lambda = \mathrm{median}(r_k/r_{k-1})`, the error
        is :math:`\approx r_{\rm last}/(1 - \hat\lambda)`. Returns ``(None, None)`` until a decaying
        rate can be measured.
        """
        if len(residuals) < _MIN_MEASURE_ITERATIONS:
            return None, None
        ratios = [residuals[i] / residuals[i - 1] for i in range(1, len(residuals)) if residuals[i - 1] > 0]
        ratios = [ratio for ratio in ratios if 0.0 < ratio < 1.0]
        if not ratios:
            return None, None
        lambda_hat = float(np.median(ratios))
        return residuals[-1] / (1.0 - lambda_hat), lambda_hat

    @staticmethod
    def _ng_accelerate(history: List[List[BaseRho]], damping: float = 1.0) -> Optional[List[BaseRho]]:
        r"""
        Ng (1974) extrapolation of the last four density-matrix iterates ``history`` (oldest first),
        returning an accelerated grid, or ``None`` if the local least-squares system is degenerate
        (near convergence). Operates on rho as an opaque complex vector, so it is atom-model
        independent; the extrapolation weights sum to one, preserving the trace normalization and the
        fixed point (all iterate differences vanish there).

        ``damping`` in (0, 1] under-relaxes the extrapolation step: the returned grid is
        :math:`\rho_0 + \mathrm{damping}\,(\rho_{\rm Ng} - \rho_0)`, which tames the overshoot of an
        aggressive extrapolation (``damping = 1`` is the full Ng step). Folding it into the weights
        keeps their sum at one.

        Reference: Ng (1974); Olson, Auer & Buchler (1986); Hubeny & Mihalas (2014), Sec. 13.3.
        """
        x3, x2, x1, x0 = history[-4], history[-3], history[-2], history[-1]
        layout = [(depth, key) for depth in range(len(x0)) for key in sorted(x0[depth].data)]
        v0 = NLTEStratifiedAtmosphere._flatten_rho_grid(x0, layout)
        v1 = NLTEStratifiedAtmosphere._flatten_rho_grid(x1, layout)
        v2 = NLTEStratifiedAtmosphere._flatten_rho_grid(x2, layout)
        v3 = NLTEStratifiedAtmosphere._flatten_rho_grid(x3, layout)

        d0 = v0 - v1
        q1 = d0 - (v1 - v2)
        q2 = d0 - (v2 - v3)
        a11 = float(q1 @ q1)
        a12 = float(q1 @ q2)
        a22 = float(q2 @ q2)
        det = a11 * a22 - a12 * a12
        if not np.isfinite(det) or det <= 1e-28 * (a11 * a22 + 1e-300):
            return None
        b1 = float(q1 @ d0)
        b2 = float(q2 @ d0)
        c1 = (b1 * a22 - b2 * a12) / det
        c2 = (a11 * b2 - a12 * b1) / det
        if not (np.isfinite(c1) and np.isfinite(c2)):
            return None
        # Damped extrapolation rho_0 + damping (rho_Ng - rho_0), with rho_Ng = (1-c1-c2) x0 + c1 x1 +
        # c2 x2. Folded into weights (which still sum to one) over [x0, x1, x2].
        c0 = 1.0 - c1 - c2
        weights = [1.0 - damping + damping * c0, damping * c1, damping * c2]
        return NLTEStratifiedAtmosphere._combine_rho_grids([x0, x1, x2], weights)

    @staticmethod
    def _flatten_rho_grid(grid: List[BaseRho], layout: List) -> np.ndarray:
        r"""
        Flatten a density-matrix grid to a real vector (real and imaginary parts) in the fixed
        ``layout`` order of ``(depth, coherence key)`` pairs, for the Ng least-squares.
        """
        vals = np.empty(2 * len(layout), dtype=np.float64)
        for idx, (depth, key) in enumerate(layout):
            value = grid[depth].data[key]
            vals[2 * idx] = value.real
            vals[2 * idx + 1] = value.imag
        return vals

    @staticmethod
    def _combine_rho_grids(grids: List[List[BaseRho]], coefficients: List[float]) -> List[BaseRho]:
        r"""
        Linear combination :math:`\sum_k c_k\,\rho_k` of density-matrix grids, coherence by
        coherence, returning a new grid with the structure of ``grids[0]``.
        """
        combined: List[BaseRho] = []
        for depth in range(len(grids[0])):
            merged = copy.deepcopy(grids[0][depth])
            for key in merged.data:
                merged.data[key] = sum(c * grid[depth].data[key] for c, grid in zip(coefficients, grids))
            combined.append(merged)
        return combined
