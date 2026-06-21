r"""
Height-stratified, self-consistent NLTE atmosphere with continuously varying physical
parameters and a full (vector) macroscopic-velocity field.

Unlike :class:`ConstantPropertySlabAtmosphere` (constant properties, imposed
:math:`J^K_Q`), this atmosphere solves the scattering :math:`J^K_Q` self-consistently and
lets *every* physical parameter vary with geometric height: temperature, absorber number
density, magnetic-field vector (magnitude *and* direction), microturbulence, Voigt
:math:`a`, and the macroscopic-velocity *vector*.

Formulation
-----------
* Geometric height ``z`` [cm] is the independent variable.  ``z[0]`` is the lower
  boundary (photosphere, or zero radiation at the limb); ``z[-1]`` is the observer side.
* The opacity scale is the local absorber number density :math:`N(z)` [cm^-3]
  (the radiative-transfer code computes per-atom coefficients and multiplies by ``N``),
  so optical depth is a physical *output*, not an imposed input.
* Transfer is solved in the observer (Eulerian) frame on the fixed ``z`` grid by DELO.
  Velocity gradients are handled by evaluating the Doppler-shifted absorption profile at
  the *local* velocity projection of each ray at each depth; this is correct provided the
  grid resolves the gradient (the per-cell line shift should stay below the line width --
  a warning is emitted otherwise).
* The line is pure scattering (collisionless).  A grey continuum is added per depth with a
  constant continuum-to-line ratio and an LTE source :math:`B(T(z))`.

Angles follow the LL04 convention (Fig. 5.9): a fixed reference frame with ``z`` along the
vertical/atmosphere normal.  A propagation direction is
:math:`\hat\Omega = (\sin\theta\cos\chi, \sin\theta\sin\chi, \cos\theta)` with
:math:`\mu = \cos\theta`; the magnetic field and the macroscopic velocity are given by
their own polar angles :math:`(\theta_B,\chi_B)` and :math:`(\theta_v,\chi_v)` in the same
frame.  The line-of-sight Doppler projection is :math:`v_{\rm los} = \hat\Omega\cdot\vec v`.

Reference: Trujillo Bueno & Manso Sainz (1999), ApJ, 516, 436 (DOI 10.1086/307107) for the
Lambda-iteration / radiation-tensor-moment scheme; generalized here to depth-dependent
properties, a vector velocity field, and the full Stokes / density-matrix case.
"""

import logging
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from numpy import real

from solrat.atom_model.base_atom_model.object.radiation_tensor import BaseRadiationTensor
from solrat.atom_model.base_atom_model.object.rho import BaseRho
from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.base_atom_model.statistical_equilibrium_equations import BaseSEE
from solrat.atom_model.model_registry import Model
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.rotations import T_K_Q
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import get_planck_BP
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.looping import FROMTO, PROJECTION
from solrat.engine.generators.nested_loops import nested_loops

# Each profile may be supplied as a scalar (constant with height), an array sampled on the
# height grid, or a callable f(z_cm) -> value.
Profile = Union[float, Sequence[float], np.ndarray, Callable[[float], float]]


def _sample_profile(value: Profile, z_cm: np.ndarray, name: str) -> np.ndarray:
    """Sample a scalar / array / callable profile onto the height grid ``z_cm``."""
    if callable(value):
        return np.array([float(value(float(zi))) for zi in z_cm], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(len(z_cm), float(arr))
    if arr.shape != z_cm.shape:
        raise ValueError(f"Profile '{name}' has length {arr.shape} but the height grid has length {z_cm.shape}.")
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
    :param continuum_to_line_ratio:  grey continuum-to-line opacity ratio (constant for now;
        the implementation leaves room for a future height-tabulated ``k_c(z)``).
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
        continuum_to_line_ratio: float = 0.0,
    ):
        z = np.asarray(height_cm, dtype=np.float64)
        if z.ndim != 1 or len(z) < 2:
            raise ValueError("height_cm must be a 1-D grid with at least 2 points.")
        if not np.all(np.diff(z) > 0):
            raise ValueError("height_cm must be strictly increasing (z[0] = lower boundary).")
        if continuum_to_line_ratio < 0:
            raise ValueError("continuum_to_line_ratio must be non-negative.")

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

        if np.any(self.temperature_K <= 0):
            raise ValueError("temperature_K must be positive everywhere.")
        if np.any(self.number_density_cm3 < 0):
            raise ValueError("number_density_cm3 must be non-negative everywhere.")

    @classmethod
    def on_uniform_grid(
        cls,
        model: Model,
        z_min_cm: float,
        z_max_cm: float,
        n_depth: int,
        **profiles: Profile,
    ) -> "StratifiedAtmosphere":
        """Convenience builder on a uniform height grid of ``n_depth`` points."""
        if n_depth < 2:
            raise ValueError("n_depth must be >= 2.")
        z = np.linspace(float(z_min_cm), float(z_max_cm), n_depth)
        return cls(model=model, height_cm=z, **profiles)

    @property
    def n_depth(self) -> int:
        return len(self.height_cm)

    def atmosphere_parameters(self, i: int, macroscopic_velocity_cm_sm1: float):
        """Build the RTE/SEE ``AtmosphereParameters`` at depth ``i`` for a given LOS velocity."""
        return self.model.AtmosphereParameters(
            model_config=self.model.config,
            magnetic_field_gauss=self.magnetic_field_gauss[i],
            temperature_K=self.temperature_K[i],
            delta_v_turbulent_cm_sm1=self.delta_v_turbulent_cm_sm1[i],
            macroscopic_velocity_cm_sm1=macroscopic_velocity_cm_sm1,
            voigt_a=self.voigt_a[i],
        )

    def magnetic_frame_angles(self, i: int) -> Angles:
        """Angles carrying only the depth-``i`` magnetic-field orientation (for J rotation)."""
        return Angles(chi_B=self.chi_B[i], theta_B=self.theta_B[i])

    def velocity_vector(self, i: int) -> np.ndarray:
        r"""Macroscopic velocity 3-vector at depth ``i`` in the fixed LL04 frame [cm/s]."""
        st = np.sin(self.theta_v[i])
        return self.velocity_cm_sm1[i] * np.array(
            [st * np.cos(self.chi_v[i]), st * np.sin(self.chi_v[i]), np.cos(self.theta_v[i])],
            dtype=np.float64,
        )


class NLTEStratifiedAtmosphere:
    r"""
    Self-consistent NLTE synthesis through a height-stratified atmosphere.

    The radiation field is propagated along a Gauss-Legendre :math:`\mu` x uniform
    :math:`\phi` quadrature of rays; the emergent Stokes distribution is projected into a
    multipole radiation tensor :math:`J^K_Q` per depth (profile-weighted in frequency,
    per ray to account for the local velocity shift), the SEE is re-solved locally, and the
    loop repeats until :math:`\max|\Delta\rho|` drops below ``tolerance``.

    :param model:  configured :class:`Model`.
    :param stratification:  :class:`StratifiedAtmosphere` height-resolved state.
    :param los_theta:  observer line-of-sight polar angle [rad] (from vertical).
    :param los_chi:  observer line-of-sight azimuth [rad].
    :param los_gamma:  observer polarization reference angle [rad] (orientation of +Q).
    :param n_mu_quadrature:  number of Gauss-Legendre points in :math:`\mu`.
    :param n_phi_quadrature:  number of uniform azimuthal samples.
    :param max_iterations:  maximum number of iterations.
    :param tolerance:  convergence threshold on :math:`\max|\Delta\rho|`.
    :param top_incident_stokes:  Stokes incident from the observer-side boundary (defaults
        to zero).  The lower boundary uses the ``initial_stokes`` passed to :meth:`forward`.
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
    ):
        if abs(np.cos(los_theta)) < 1e-4:
            raise ValueError("Observer cos(theta) too close to zero (tangential view).")
        if n_mu_quadrature < 1 or n_phi_quadrature < 1:
            raise ValueError("Need at least one mu and one phi quadrature point.")

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

        # Diagnostics populated by forward()
        self.rho_grid: Optional[List[BaseRho]] = None
        self.tau_grid: Optional[np.ndarray] = None
        self.iterations_used: Optional[int] = None
        self.final_residual: Optional[float] = None
        self.residual_history: List[float] = []

        # Fixed frequency partition for the J-reconstruction profile weights (set in forward()).
        self._recon_transition_ids: List[str] = []
        self._recon_centers: Optional[np.ndarray] = None
        self._recon_nearest: Optional[np.ndarray] = None

    @log_method
    def forward(self, initial_stokes: Stokes) -> Stokes:
        r"""
        Run the NLTE loop and return the emergent Stokes vector along the observer ray.

        :param initial_stokes:  Stokes incident at the lower boundary ``z[0]``.
        """
        nu = initial_stokes.nu
        strat = self.stratification
        n_z = strat.n_depth
        z = strat.height_cm
        mu_obs = float(np.cos(self.los_theta))

        self.residual_history = []

        see: BaseSEE = self.model.StatisticalEquilibriumEquations.from_model_config(self.model.config)
        rte: BaseRTE = self.model.RadiativeTransferEquations.from_model_config(self.model.config, nu=nu)
        # Opacity is carried by the per-depth number density: set rte.N = N(z_i) per call. N is the
        # final factor applied after the cached operator, so the operator cache stays valid.
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

        # 2. Per-depth line-core opacity along the observer ray (absolute, includes N), and the
        #    grey continuum opacity from the constant continuum-to-line ratio.
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
        if float(np.max(eta_peak)) <= 0:
            raise ValueError(
                "Line opacity along the observer ray is zero for the initial guess. "
                "Check that the frequency grid covers the transition and N(z) > 0."
            )
        k_c_per_z = strat.continuum_to_line_ratio * eta_peak  # [n_z]

        # Observer-ray line optical depth (diagnostic): tau = int eta dz / |mu_obs|.
        d_tau = 0.5 * (eta_peak[1:] + eta_peak[:-1]) * np.diff(z) / abs(mu_obs)
        self.tau_grid = np.concatenate([[0.0], np.cumsum(d_tau)])

        # Frequency partition for the J-reconstruction profile weights: nearest transition per
        # grid point. Depends only on nu, so compute it once for the whole run.
        recon_transitions = list(
            self.model.RadiationTensor.from_model_config(self.model.config).transition_registry.transitions.values()
        )
        self._recon_transition_ids = [t.transition_id for t in recon_transitions]
        self._recon_centers = np.array(
            [t.get_mean_transition_frequency_sm1() for t in recon_transitions], dtype=np.float64
        )
        self._recon_nearest = np.argmin(np.abs(nu[:, np.newaxis] - self._recon_centers[np.newaxis, :]), axis=1)

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
                    Angles(chi=ray["chi"], theta=ray["theta"], gamma=0.0, chi_B=strat.chi_B[i], theta_B=strat.theta_B[i])
                    for i in range(n_z)
                ]
            )

        bottom_bc = initial_stokes
        top_bc = self.top_incident_stokes if self.top_incident_stokes is not None else Stokes.from_zeros(nu_sm1=nu)

        # 4. Iterate.
        for iteration in range(self.max_iterations):
            stokes_per_ray: List[np.ndarray] = []
            eta_I_per_ray: List[np.ndarray] = []
            for r in range(len(rays)):
                stokes_z, eta_I_z = self._propagate_ray(
                    rho_grid=rho_grid, z=z, mu_n=rays[r]["mu"], rte=rte,
                    params_per_z=ray_params[r], angles_per_z=ray_angles[r],
                    number_density=N, k_c_per_z=k_c_per_z, bp_per_z=bp_per_z,
                    bottom_bc=bottom_bc, top_bc=top_bc,
                )  # fmt: skip
                stokes_per_ray.append(stokes_z)
                eta_I_per_ray.append(eta_I_z)

            new_rho_grid: List[BaseRho] = []
            for i in range(n_z):
                radiation_tensor_i = self._reconstruct_radiation_tensor(
                    rays=rays, stokes_per_ray=stokes_per_ray, eta_I_per_ray=eta_I_per_ray,
                    i_z=i, t_conj_per_ray=t_conj_per_ray, nu=nu,
                )  # fmt: skip
                see.fill_all_equations(
                    atmosphere_parameters=see_params[i],
                    radiation_tensor_in_magnetic_frame=radiation_tensor_i.rotate_to_magnetic_frame(angles=b_angles[i]),
                )
                new_rho_grid.append(see.get_solution())

            residual = self._rho_grid_diff(rho_grid, new_rho_grid)
            rho_grid = new_rho_grid
            self.iterations_used = iteration + 1
            self.final_residual = residual
            self.residual_history.append(residual)
            logging.warning(f"NLTE (stratified) iteration {iteration}: residual = {residual:.3e}")
            if residual < self.tolerance:
                break

        self.rho_grid = rho_grid

        # 5. Final propagation along the observer ray.
        obs_params = [strat.atmosphere_parameters(i, self._project(v_vectors[i], omega_obs)) for i in range(n_z)]
        observer_stokes_z, _ = self._propagate_ray(
            rho_grid=rho_grid, z=z, mu_n=mu_obs, rte=rte,
            params_per_z=obs_params, angles_per_z=obs_angles,
            number_density=N, k_c_per_z=k_c_per_z, bp_per_z=bp_per_z,
            bottom_bc=bottom_bc, top_bc=top_bc,
        )  # fmt: skip
        emergent = observer_stokes_z[-1] if mu_obs > 0 else observer_stokes_z[0]

        if hasattr(rte, "clear_operator_cache"):
            rte.clear_operator_cache()

        return Stokes(
            nu=nu,
            I=real(emergent[0]),
            Q=real(emergent[1]),
            U=real(emergent[2]),
            V=real(emergent[3]),
        )

    # ------------------------------------------------------------------ geometry / velocity

    @staticmethod
    def _ray_direction(theta: float, chi: float) -> np.ndarray:
        r"""Propagation unit vector :math:`\hat\Omega` in the fixed LL04 frame (z = vertical)."""
        st = np.sin(theta)
        return np.array([st * np.cos(chi), st * np.sin(chi), np.cos(theta)], dtype=np.float64)

    @staticmethod
    def _project(v_vector: np.ndarray, omega: np.ndarray) -> float:
        r"""
        Line-of-sight velocity projection feeding the absorption profile,
        :math:`v_{\rm los} = -\,\hat\Omega\cdot\vec v`, with :math:`\hat\Omega` the photon
        propagation (toward-observer) direction.

        The minus sign makes this consistent with the bundled profile convention
        (:math:`\nu = \nu_i (1 - v_{\rm los}/c)`, i.e. positive :math:`v_{\rm los}` = redshift):
        plasma moving *toward* the observer (a velocity component along :math:`\hat\Omega`)
        gives :math:`v_{\rm los} < 0` and therefore a blueshift, as required physically. So
        :math:`v_{\rm los}` is the projection onto the into-the-medium direction
        (receding-positive).
        """
        return -float(np.dot(v_vector, omega))

    def _build_quadrature_rays(self) -> List[dict]:
        r"""
        Gauss-Legendre :math:`\mu` x uniform :math:`\phi` quadrature.  Each ray weight already
        includes the :math:`1/(4\pi)` normalization of :math:`J^K_Q`.
        """
        mus, mu_weights = np.polynomial.legendre.leggauss(self.n_mu_quadrature)
        phi_grid = np.linspace(0.0, 2 * np.pi, self.n_phi_quadrature, endpoint=False)
        phi_weight_each = 2 * np.pi / self.n_phi_quadrature

        rays: List[dict] = []
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

    def _t_conj_for_ray(self, ray: dict) -> dict:
        r"""Pre-compute :math:`T^{K*}_Q(i, \Omega)` for one ray (gamma = 0 for quadrature rays)."""
        kq_pairs = list(nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")))
        return {
            (int(K), int(Q)): np.array(
                [
                    T_K_Q(
                        K=int(K), Q=int(Q), stokes_component_index=k,
                        chi=ray["chi"], theta=ray["theta"], gamma=0.0,
                    ).conjugate()
                    for k in range(4)
                ],
                dtype=np.complex128,
            )  # fmt: skip
            for (K, Q) in kq_pairs
        }

    def _warn_if_velocity_underresolved(
        self, rays: List[dict], v_vectors: List[np.ndarray], thermal_v_per_z: np.ndarray, nu: np.ndarray
    ) -> None:
        """
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
        params_per_z: list,
        angles_per_z: List[Angles],
        number_density: np.ndarray,
        k_c_per_z: np.ndarray,
        bp_per_z: List[np.ndarray],
        bottom_bc: Stokes,
        top_bc: Stokes,
    ):
        r"""
        DELO-propagate Stokes through the ``z`` grid along one ray.

        Returns ``(stokes[n_z, 4, n_nu], eta_I[n_z, n_nu])`` where ``eta_I`` is the line
        absorption (used as the per-(ray, depth) profile weight in the J reconstruction).
        ``z[0]`` is the lower boundary, ``z[-1]`` the observer side.  ``mu_n > 0`` propagates
        upward (boundary at ``z[0]``); ``mu_n < 0`` downward (boundary at ``z[-1]``).
        """
        n_z = len(z)
        nu = bottom_bc.nu
        n_nu = len(nu)
        if abs(mu_n) < 1e-6:
            raise ValueError(f"Quadrature mu = {mu_n} is too close to tangential.")

        # Coefficients at every depth (line, absolute via rte.N, plus grey continuum).
        K_per_z = []
        eps_per_z = []
        eta_I = np.zeros((n_z, n_nu))
        for i in range(n_z):
            rte.N = float(number_density[i])
            rtc = rte.calculate_all_coefficients(
                atmosphere_parameters=params_per_z[i], angles=angles_per_z[i], rho=rho_grid[i]
            )
            K = rtc.K_z()  # [n_nu, 4, 4]
            eps = rtc.epsilon_z()[:, :, 0]  # [n_nu, 4]
            for k in range(4):
                K[:, k, k] += k_c_per_z[i]
            eps[:, 0] += k_c_per_z[i] * bp_per_z[i]
            K_per_z.append(K)
            eps_per_z.append(eps)
            eta_I[i] = np.real(rtc.get_eta_I())

        stokes = np.zeros((n_z, 4, n_nu), dtype=np.complex128)
        if mu_n > 0:
            stokes[0] = np.stack([bottom_bc.I, bottom_bc.Q, bottom_bc.U, bottom_bc.V])
            for i in range(0, n_z - 1):
                ds = (z[i + 1] - z[i]) / abs(mu_n)
                stokes[i + 1] = self._delo_matrix_step(K_per_z[i], eps_per_z[i], stokes[i], ds)
        else:
            stokes[-1] = np.stack([top_bc.I, top_bc.Q, top_bc.U, top_bc.V])
            for i in range(n_z - 1, 0, -1):
                ds = (z[i] - z[i - 1]) / abs(mu_n)
                stokes[i - 1] = self._delo_matrix_step(K_per_z[i], eps_per_z[i], stokes[i], ds)
        return stokes, eta_I

    @staticmethod
    def _delo_matrix_step(K: np.ndarray, epsilon: np.ndarray, current_stokes: np.ndarray, ds: float) -> np.ndarray:
        r"""
        One DELO step over physical length ``ds``:
        :math:`S = K^{-1}\epsilon,\; \mathrm{Stokes}(s+ds) = S + e^{-K\,ds}(\mathrm{Stokes}(s) - S)`,
        batched over frequency.  ``K`` is ``[n_nu, 4, 4]``, ``epsilon`` ``[n_nu, 4]``,
        ``current_stokes`` ``[4, n_nu]``.
        """
        cond = np.linalg.cond(K)
        well = cond < 1e12
        S = np.empty_like(epsilon)  # [n_nu, 4]
        if np.any(well):
            S[well] = np.linalg.solve(K[well], epsilon[well][:, :, np.newaxis])[:, :, 0]
        if not np.all(well):
            S[~well] = np.einsum("nij,nj->ni", np.linalg.pinv(K[~well]), epsilon[~well])

        lam, V = np.linalg.eig(-K * ds)  # lam [n_nu, 4], V [n_nu, 4, 4]
        expM = np.real(V @ (np.exp(lam)[:, :, np.newaxis] * np.linalg.inv(V)))  # [n_nu, 4, 4]

        current = current_stokes.T[:, :, np.newaxis]  # [n_nu, 4, 1]
        new = S[:, :, np.newaxis] + np.einsum("nij,njk->nik", expM, current - S[:, :, np.newaxis])
        return new[:, :, 0].T  # [4, n_nu]

    # ------------------------------------------------------------------ radiation tensor

    def _reconstruct_radiation_tensor(
        self,
        rays: List[dict],
        stokes_per_ray: List[np.ndarray],
        eta_I_per_ray: List[np.ndarray],
        i_z: int,
        t_conj_per_ray: List[dict],
        nu: np.ndarray,
    ) -> BaseRadiationTensor:
        r"""
        Reconstruct :math:`J^K_Q` per transition at depth ``i_z``:

        .. math::

            J^K_Q = \sum_n w_n \sum_i T^{K*}_Q(i, \Omega_n) \int d\nu\, \phi_n(\nu)\, S_i(\nu, \Omega_n)

        The frequency weight :math:`\phi_n` is the *per-ray* line absorption profile at this
        depth (so the local velocity shift seen along each ray is accounted for in complete
        frequency redistribution).  Reduces to Trujillo Bueno & Manso Sainz (1999) eqs.
        (10)-(11) for the axisymmetric K=0,2, Q=0 case.
        """
        rad_tens = self.model.RadiationTensor.from_model_config(self.model.config)
        kq_pairs = list(nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")))

        accumulator: dict = {}
        for r, ray in enumerate(rays):
            weights = self._build_profile_weights(nu=nu, eta_I=np.maximum(eta_I_per_ray[r][i_z], 0.0))
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

    def _build_profile_weights(self, nu: np.ndarray, eta_I: np.ndarray) -> dict:
        r"""
        Normalized line-absorption profile weight :math:`\hat\phi(\nu)` per transition (sums to 1),
        partitioned by nearest transition center for well-separated lines (flat-spectrum).

        Uses the fixed ``self._recon_*`` partition precomputed in :meth:`forward` (depends only
        on ``nu``), so the per-(ray, depth) call is just a masked normalization.
        """
        weights = {}
        for k, transition_id in enumerate(self._recon_transition_ids):
            w = np.where(self._recon_nearest == k, eta_I, 0.0)
            total = float(w.sum())
            if total <= 0.0:
                w = np.zeros_like(nu)
                w[int(np.argmin(np.abs(nu - self._recon_centers[k])))] = 1.0
            else:
                w = w / total
            weights[transition_id] = w
        return weights

    @staticmethod
    def _rho_grid_diff(rho_grid_old: List[BaseRho], rho_grid_new: List[BaseRho]) -> float:
        r"""Maximum :math:`|\rho_{\rm new} - \rho_{\rm old}|` over the grid and coherences."""
        max_diff = 0.0
        for rho_old, rho_new in zip(rho_grid_old, rho_grid_new):
            for key, val_new in rho_new.data.items():
                d = abs(val_new - rho_old.data.get(key, 0.0))
                if d > max_diff:
                    max_diff = d
        return float(max_diff)
