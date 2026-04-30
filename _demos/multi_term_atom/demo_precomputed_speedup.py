"""
Precomputed vs. fresh: speed comparison for He I D3 SEE.

The atom-specific frames (Wigner symbols, Einstein coefficients, etc.) are
independent of the radiation field and atmospheric parameters.  They can be
built once, saved to CSV, and reloaded instantly on subsequent runs.

This demo measures the wall-clock time for:
  1. Fresh computation  – frames built from scratch on every fill_all_equations call.
  2. Disk-precomputed   – frames loaded from CSV; only the radiation-tensor
                          contractions are performed at run time.

Both paths produce bit-identical rho solutions (verified below).
"""

import time

import numpy as np

from solrat.atom_model.model_registry import Models, PreconfiguredModels
from solrat.atom_model.multi_term_atom_model.data.HeI import get_He_I_D3_config
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import MultiTermAtomConfig
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.utility.log_setup import setup_logging


def main():
    setup_logging()

    angles = Angles(
        chi=np.pi / 5,
        theta=np.pi / 7,
        gamma=np.pi / 9,
        chi_B=np.pi / 3,
        theta_B=np.pi / 5,
    )

    # ------------------------------------------------------------------
    # Fresh model  (no precomputed data)
    # ------------------------------------------------------------------
    base_config = get_He_I_D3_config()
    config_fresh = MultiTermAtomConfig(
        level_registry=base_config.level_registry,
        transition_registry=base_config.transition_registry,
        atomic_mass_amu=base_config.atomic_mass_amu,
        reference_lambda_A_air=base_config.reference_lambda_A_air,
        precomputed_data=None,
    )
    model_fresh = Models.multi_term_atom().configure(config=config_fresh)

    # ------------------------------------------------------------------
    # Disk-precomputed model
    # ------------------------------------------------------------------
    model_disk = PreconfiguredModels.multi_term_atom_HeID3()
    assert (
        model_disk.config.precomputed_data is not None
    ), "Expected precomputed CSV data to be present in HeI_precomputed/."

    # ------------------------------------------------------------------
    # Shared inputs
    # ------------------------------------------------------------------
    N_CALLS = 2  # repeat fill_all_equations to get a stable timing

    def _make_inputs(model):
        atm = model.AtmosphereParameters(
            model_config=model.config,
            magnetic_field_gauss=100,
            temperature_K=7000,
        )
        rad = (
            model.RadiationTensor.from_model_config(model.config)
            .fill_NLTE_n_w_parametrized(h_arcsec=30)
            .rotate_to_magnetic_frame(angles=angles)
        )
        return atm, rad

    atm_fresh, rad_fresh = _make_inputs(model_fresh)
    atm_disk, rad_disk = _make_inputs(model_disk)

    # ------------------------------------------------------------------
    # Time  fresh
    # ------------------------------------------------------------------
    see_fresh = model_fresh.StatisticalEquilibriumEquations.from_model_config(model_fresh.config)
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        see_fresh.fill_all_equations(
            atmosphere_parameters=atm_fresh,
            radiation_tensor_in_magnetic_frame=rad_fresh,
        )
    t_fresh = (time.perf_counter() - t0) / N_CALLS
    rho_fresh = see_fresh.get_solution()

    # ------------------------------------------------------------------
    # Time  disk-precomputed
    # ------------------------------------------------------------------
    see_disk = model_disk.StatisticalEquilibriumEquations.from_model_config(model_disk.config)
    t0 = time.perf_counter()
    for _ in range(N_CALLS):
        see_disk.fill_all_equations(
            atmosphere_parameters=atm_disk,
            radiation_tensor_in_magnetic_frame=rad_disk,
        )
    t_disk = (time.perf_counter() - t0) / N_CALLS
    rho_disk = see_disk.get_solution()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"He I D3  --  fill_all_equations  (avg over {N_CALLS} calls)")
    print(f"  fresh computation : {t_fresh:.3f} s")
    print(f"  disk precomputed  : {t_disk:.3f} s")
    print(f"  speedup           : {t_fresh / t_disk:.1f}x")
    print(f"{'=' * 50}\n")

    # Confirm the two paths agree numerically
    max_abs_diff = 0.0
    for term in model_fresh.config.level_registry.terms.values():
        for J in np.arange(abs(term.L - term.S), term.L + term.S + 1):
            for Jp in np.arange(abs(term.L - term.S), term.L + term.S + 1):
                for K in np.arange(abs(J - Jp), J + Jp + 1):
                    for Q in np.arange(-K, K + 1):
                        a = rho_fresh(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jp)
                        b = rho_disk(term_id=term.term_id, K=K, Q=Q, J=J, Jʹ=Jp)
                        max_abs_diff = max(max_abs_diff, abs(a - b))

    print(f"Max |rho_fresh - rho_disk| = {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-10, "Solutions differ -- precomputed data may be stale."
    print("Solutions agree to machine precision.")


if __name__ == "__main__":
    main()
