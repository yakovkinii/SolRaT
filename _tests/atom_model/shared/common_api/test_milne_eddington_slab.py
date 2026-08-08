import unittest

import numpy as np

from _demos.general.demo_unno_rachkovsky_ME import (
    build_normal_triplet_lte_model,
    unno_rachkovsky_emergent_stokes,
    zeeman_triplet_line_coefficients,
)
from solrat.atom_model.shared.common_api.milne_eddington_slab import MilneEddingtonSlabAtmosphere
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.constants import c_cm_sm1
from solrat.atom_model.shared.utility.functions import get_frequencies_from_air_wavelength_range, nu_larmor
from solrat.atom_model.shared.utility.log_setup import setup_logging


class TestMilneEddingtonSlabVsUnnoRachkovsky(unittest.TestCase):
    r"""
    The Milne-Eddington slab (SolRaT numeric) must reproduce the analytic Unno-Rachkovsky solution
    for a normal Zeeman triplet: the normalization-free propagation-matrix coefficient ratios and the
    emergent Stokes profiles agree, in particular the LL04 Stokes-V sign (eq. 5.36).
    """

    def _synthesize(self, magnetic_field_gauss: float) -> dict:
        r"""
        Synthesize a ^1S_0 -> ^1P_1 LTE line (normal Zeeman triplet, g_u = 1) with the
        Milne-Eddington slab and build the analytic Unno-Rachkovsky solution from the same
        parameters. Returns the SolRaT and analytic outputs plus the reduced-frequency grid.

        :param magnetic_field_gauss: magnetic field strength [G].
        :return: dict with SolRaT Stokes, analytic (I, Q, U, V), the RTE coefficients, and the
            reduced-frequency grid and geometry needed for the coefficient-ratio comparison.
        """
        setup_logging()
        voigt_a = 0.05
        theta_B, chi_B = np.deg2rad(60.0), np.deg2rad(0.0)
        eta_0, source_0, source_1 = 10.0, 1.0, 3.0

        model, nu0, reference_lambda_A_air = build_normal_triplet_lte_model()
        nu = get_frequencies_from_air_wavelength_range(
            lower_wavelength_A=reference_lambda_A_air - 0.35,
            upper_wavelength_A=reference_lambda_A_air + 0.35,
            step_A=1e-3,
        )
        atmosphere_parameters = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=magnetic_field_gauss,
            temperature_K=6000.0,
            delta_v_turbulent_cm_sm1=2.0e5,
            voigt_a=voigt_a,
        )
        angles = Angles(chi=0.0, theta=0.0, gamma=0.0, chi_B=chi_B, theta_B=theta_B)
        slab = MilneEddingtonSlabAtmosphere(
            model=model,
            radiation_tensor=model.RadiationTensor(),
            atmosphere_parameters=atmosphere_parameters,
            angles=angles,
            line_to_continuum_ratio=eta_0,
            source_gradient=source_1,
            source_surface=source_0,
        )
        solrat_stokes = slab.forward(initial_stokes=Stokes.from_zeros(nu_sm1=nu))

        delta_nu_D = nu0 * atmosphere_parameters.delta_v_thermal_cm_sm1 / c_cm_sm1
        v = (nu - nu0) / delta_nu_D
        v_B = float(nu_larmor(np.array(magnetic_field_gauss))) / delta_nu_D  # g_u = 1 for ^1P_1
        analytic = unno_rachkovsky_emergent_stokes(
            v=v, a=voigt_a, v_B=v_B, theta_B=theta_B, chi_B=chi_B, eta_0=eta_0, mu=1.0,
            source_0=source_0, source_1=source_1, normalization="max",
        )  # fmt: skip
        return {
            "solrat": solrat_stokes, "analytic": analytic, "rtc": slab.rtc,
            "v": v, "v_B": v_B, "theta_B": theta_B, "chi_B": chi_B, "voigt_a": voigt_a,
        }  # fmt: skip

    def test_coefficient_ratios_match_normal_triplet(self):
        r"""
        The normalization-free ratios eta_Q/eta_I, eta_V/eta_I, rho_V/eta_I from SolRaT's RTE match
        the analytic normal-triplet expressions over the line core (independent of the opacity scale).
        """
        out = self._synthesize(1500.0)
        rtc, v = out["rtc"], out["v"]
        eta_I = rtc.get_eta_I()
        core = np.abs(eta_I) > 0.05 * np.max(np.abs(eta_I))
        eta_I_line, eta_Q, _, eta_V, _, _, rho_V = zeeman_triplet_line_coefficients(
            v, out["voigt_a"], out["v_B"], out["theta_B"], out["chi_B"]
        )
        # Conservative starting tolerance (catches a Stokes-V sign flip); tighten after a local run.
        np.testing.assert_allclose(rtc.get_eta_Q()[core] / eta_I[core], eta_Q[core] / eta_I_line[core], atol=1e-5)
        np.testing.assert_allclose(rtc.get_eta_V()[core] / eta_I[core], eta_V[core] / eta_I_line[core], atol=1e-5)
        np.testing.assert_allclose(rtc.get_rho_V()[core] / eta_I[core], rho_V[core] / eta_I_line[core], atol=1e-5)

    def test_emergent_stokes_match_analytic(self):
        r"""
        The emergent I, Q/I, U/I, V/I from the Milne-Eddington slab match the analytic
        Unno-Rachkovsky solution built from the same physical parameters.
        """
        out = self._synthesize(1500.0)
        solrat = out["solrat"]
        stokes_I_a, stokes_Q_a, stokes_U_a, stokes_V_a = out["analytic"]
        np.testing.assert_allclose(solrat.I, stokes_I_a, rtol=1e-5)
        np.testing.assert_allclose(solrat.Q / solrat.I, stokes_Q_a / stokes_I_a, atol=1e-5)
        np.testing.assert_allclose(solrat.U / solrat.I, stokes_U_a / stokes_I_a, atol=1e-5)
        np.testing.assert_allclose(solrat.V / solrat.I, stokes_V_a / stokes_I_a, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
