import unittest

import numpy as np
from numpy import exp

from solrat.atom_model.model_registry import PreconfiguredModels
from solrat.atom_model.multi_level_atom_model.object.collisions import ParametrizedCollisions
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    NLTEStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, kB_erg_Km1
from solrat.atom_model.shared.utility.log_setup import setup_logging


def _c_ul_for_epsilon(epsilon, transition, temperature_K):
    r"""Collisional de-excitation rate :math:`C_{ul}` giving a two-level photon destruction probability
    ``epsilon`` (TB1999 Sec. 2)."""
    delta_e_erg = (transition.level_upper.energy_cmm1 - transition.level_lower.energy_cmm1) * h_erg_s * c_cm_sm1
    stimulated_correction = 1.0 - exp(-delta_e_erg / (kB_erg_Km1 * temperature_K))
    return epsilon / (1.0 - epsilon) * transition.einstein_a_ul / stimulated_correction


def _log_depth_grid(z_max_cm, n_depth, min_fraction=1e-7):
    r"""Height grid with the depth below the surface log-spaced, so the surface optical-depth decades
    are resolved. ``z[0]`` is the lower boundary, ``z[-1]`` the observer surface."""
    depth_below_surface = np.logspace(np.log10(z_max_cm * min_fraction), np.log10(z_max_cm), n_depth)
    return np.sort(z_max_cm - depth_below_surface)


def _symmetric_frequency_grid(transition, delta_v_thermal_cm_sm1, n_half=3):
    r"""Frequency grid symmetric about line center (:math:`\nu_0` at the middle index), spanning
    :math:`\pm 4` Doppler widths, so a static isothermal profile is exactly mirror-symmetric on it."""
    nu0 = transition.get_mean_transition_frequency_sm1()
    delta_nu_D = nu0 * delta_v_thermal_cm_sm1 / c_cm_sm1
    return nu0 + np.linspace(-4.0, 4.0, 2 * n_half + 1) * delta_nu_D


class TestNLTEStratifiedTB1999(unittest.TestCase):
    r"""
    TB1999 resonance-polarization benchmark (a J=0 -> J=1 two-level atom in an isothermal, self-emitting
    slab; no continuum, no field, epsilon = 1e-2). Each observable is checked two ways: a loose
    comparison against the tabulated TB1999 value (right sign and order; the coarse grids used for speed
    under-resolve the angular quadrature and the surface, reaching roughly half the tabulated value), and
    a tight regression against the current computed value. The demos do the fully-resolved quantitative
    comparison.
    """

    def _solve(self, mu_observer, n_depth):
        r"""Solve the tiny TB1999 slab along a line of sight ``mu_observer``; return
        (atmosphere, emergent, nu, transition). A fixed iteration count is used (tolerance = 0):
        Lambda-iteration's per-step change is small while the alignment is still building up, so a
        residual threshold would stop before it does."""
        setup_logging()
        temperature_K = 6000.0
        collisions = ParametrizedCollisions()
        model = PreconfiguredModels.multi_level_atom_mock(collisions=collisions)
        transition = next(iter(model.config.transition_registry.transitions.values()))
        collisions.set_deexcitation_rate(transition.transition_id, _c_ul_for_epsilon(1e-2, transition, temperature_K))

        params = model.AtmosphereParameters(
            model_config=model.config, magnetic_field_gauss=0.0, temperature_K=temperature_K
        )
        nu = _symmetric_frequency_grid(transition, params.delta_v_thermal_cm_sm1)
        stratification = StratifiedAtmosphere(
            model=model,
            height_cm=_log_depth_grid(1000e5, n_depth),
            temperature_K=temperature_K,
            number_density_cm3=1.0e11,
            magnetic_field_gauss=0.0,
            velocity_cm_sm1=0.0,
            delta_v_turbulent_cm_sm1=0.0,
            voigt_a=0.0,
            continuum_to_line_ratio=0.0,
        )
        atmosphere = NLTEStratifiedAtmosphere(
            model=model,
            stratification=stratification,
            los_theta=float(np.arccos(mu_observer)),
            los_chi=0.0,
            los_gamma=0.0,
            n_mu_quadrature=6,
            n_phi_quadrature=3,
            max_iterations=25,
            tolerance=0.0,
            ng_acceleration=True,
            ng_damping=0.7,
        )
        emergent = atmosphere.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))
        return atmosphere, emergent, nu, transition

    def test_tangential_surface_alignment_and_qi(self):
        r"""
        Tangential (mu = 0) solve: the upper-level alignment rho^2_0/rho^0_0 at the surface (tau -> 0)
        and the emergent line-center Q/I. Both are compared against TB1999 Table 4 (epsilon = 1e-2:
        alignment 0.05666, Q/I -6.132%) loosely and pinned as regressions. The tangential emergent is
        the surface source function, so Q/I equals the surface alignment times the fixed J=0 -> J=1
        geometric factor (~ -108%), a resolution-independent relation checked directly.
        """
        atmosphere, emergent, nu, transition = self._solve(mu_observer=0.0, n_depth=16)
        upper_level_id = transition.level_upper.level_id
        rho_surface = atmosphere.rho_grid[-1]
        alignment = np.real(rho_surface(K=2, Q=0, level_id=upper_level_id)) / np.real(
            rho_surface(K=0, Q=0, level_id=upper_level_id)
        )
        qi_percent = 100.0 * np.real(emergent.Q[len(nu) // 2] / emergent.I[len(nu) // 2])

        # Loosely vs TB1999 Table 4 (right sign and order; the coarse angular grid reaches ~half).
        assert 0.3 < alignment / 0.05666 < 1.1
        assert 0.3 < qi_percent / -6.132 < 1.1
        # Emergent = surface source function: Q/I is the alignment times the geometric factor.
        assert -115.0 < qi_percent / alignment < -100.0
        # Regression against the current computed values.
        assert np.isclose(alignment, 0.0298734, rtol=1e-3)
        assert np.isclose(qi_percent, -3.20237, rtol=1e-3)

    def test_inclined_emergent_qi_profile(self):
        r"""
        Inclined (mu = 0.1) emergent Q/I profile (TB1999 Fig. 10, delta2 = 0). The static isothermal
        slab makes the profile symmetric about line center, and the line core is negatively polarized.
        The line-center Q/I is compared against the digitized TB1999 Fig. 10 value (-3.40%) loosely (the
        coarse depth under-resolves the core, reaching ~40%) and pinned as a regression.
        """
        atmosphere, emergent, nu, transition = self._solve(mu_observer=0.1, n_depth=24)
        qi_percent = 100.0 * np.real(emergent.Q / emergent.I)
        center = len(nu) // 2

        assert np.all(np.isfinite(qi_percent))
        assert np.allclose(qi_percent, qi_percent[::-1], atol=1e-3, rtol=1e-4)  # symmetric about line center
        assert qi_percent[center] < 0  # negatively polarized line core
        # Loosely vs TB1999 Fig. 10 line center (right sign and order; the coarse depth reaches ~40%).
        assert 0.2 < qi_percent[center] / -3.40158 < 1.1
        # Regression against the current computed value.
        assert np.isclose(qi_percent[center], -1.33128, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
