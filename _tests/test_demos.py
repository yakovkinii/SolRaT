import importlib
import logging
import unittest

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

DEMO_MODULES = [
    "_demos.start_here.demo_basic_stokes_profile_synthesis",
    "_demos.multi_term_atom.demo_constant_property_slab_HeI_D3",
    "_demos.multi_term_atom.demo_hanle_effect",
    "_demos.multi_term_atom.demo_precomputed_speedup",
    "_demos.multi_term_atom.demo_radiative_transfer_equations_D3",
    "_demos.multi_term_atom.demo_radiative_transfer_equations_D3_parameter_sweep",
    "_demos.multi_term_atom.demo_radiative_transfer_equations_resonance",
    "_demos.multi_term_atom_lte.demo_constant_property_slab_MnI",
    "_demos.multi_term_atom_lte.demo_constant_property_slab_MnI_FeI_NiI",
    "_demos.multi_term_atom_lte.demo_stokes_vs_B_lte_MnI",
    "_demos.general.demo_collisional_depolarization",
    "_demos.general.demo_collisions_lte_limit",
    "_demos.general.demo_DELO",
    "_demos.general.demo_delo_constant_vs_linear",
    "_demos.general.demo_hazel_comparison_HeID3",
    "_demos.general.demo_multi_term_vs_multi_level_divergence",
    "_demos.general.demo_multi_term_vs_multi_level_S0",
    "_demos.general.demo_multi_term_vs_multi_level_S0_lte_nlte",
    "_demos.general.demo_nlte_multi_term_vs_multi_level_selfconsistent",
    "_demos.general.demo_NLTE_n_w_parametrized",
    "_demos.general.demo_nlte_stratified_atmosphere",
    "_demos.general.demo_nlte_thermalization_sqrt_epsilon",
    "_demos.general.demo_nlte_TM99_convergence_history",
    "_demos.general.demo_nlte_TM99_resonance_polarization",
    "_demos.general.demo_nlte_TM99_resonance_polarization_mu01",
    "_demos.general.demo_paschen_back",
    "_demos.general.demo_paschen_back_FeI5434",
    "_demos.general.demo_performance_scaling",
    "_demos.general.demo_single_scattering_polarization",
    "_demos.general.demo_unno_rachkovsky_ME",
    "_demos.general.demo_voigt_profile",
    "_demos.general.demo_zeeman_pattern",
]


class TestDemos(unittest.TestCase):
    def test_demos_are_runnable(self):
        for module_name in DEMO_MODULES:
            with self.subTest(module=module_name):
                logging.info("Running demo %s", module_name)
                module = importlib.import_module(module_name)
                assert hasattr(module, "main"), f"{module_name} does not expose main()"
                module.main()
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
