# Manuscript Figure and Benchmark Demos

This file maps manuscript figures and reproducibility checks to the public demo scripts in
`_demos/`.
## Main Figures

| Manuscript item | Demo source |
| --- | --- |
| Fig. 2: He I D3 Stokes profiles | `_demos/multi_term_atom/demo_constant_property_slab_HeI_D3.py` |
| Fig. 3: TM99 alignment benchmark | `_demos/general/demo_nlte_TM99_resonance_polarization.py` |
| Fig. 4: second-order Zeeman comparison | `_demos/general/demo_multi_term_vs_multi_level_divergence.py` |

## Appendix Figures

| Manuscript item | Demo source |
| --- | --- |
| Fig. B.1: Paschen-Back sublevel energies | `_demos/general/demo_paschen_back.py` |
| Fig. F.1: He I D3 comparison with HAZEL2 | `_demos/general/demo_hazel_comparison_HeID3.py` |
| Fig. F.2: Fe I, Ni I, Mn I photospheric triplet | `_demos/multi_term_atom_lte/demo_constant_property_slab_MnI_FeI_NiI.py` |
| Fig. F.3: LTE and non-LTE multi-term/multi-level comparison | `_demos/general/demo_multi_term_vs_multi_level_divergence.py` |
| Fig. F.4: AH65 sqrt-epsilon thermalization benchmark | `_demos/general/demo_nlte_thermalization_sqrt_epsilon.py` |
| Fig. G.1: Hanle depolarization benchmark | `_demos/multi_term_atom/demo_hanle_effect.py` |
| Fig. G.2: stratified prescribed-JKQ vs Unno-Rachkovsky benchmark | `_demos/general/demo_unno_rachkovsky_ME.py` |
| Fig. G.3: S=0 multi-term vs multi-level cross-check | `_demos/general/demo_multi_term_vs_multi_level_S0.py` |

## Additional Reproducibility Checks

| Check | Demo source |
| --- | --- |
| Voigt approximation against SciPy Faddeeva function | `_demos/general/demo_voigt_profile.py` |
| Single-scattering angular polarization | `_demos/general/demo_single_scattering_polarization.py` |
| Collisional detailed-balance LTE limit | `_demos/general/demo_collisions_lte_limit.py` |
| Zeeman-pattern diagnostic against the linear-Zeeman limit | `_demos/general/demo_zeeman_pattern.py` |

`_demos/general/demo_multi_term_vs_multi_level_divergence.py` returns two figures: the
second-order Zeeman comparison and the LTE/non-LTE scattering comparison.
