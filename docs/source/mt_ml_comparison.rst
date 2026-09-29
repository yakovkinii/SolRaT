Multi-Term vs Multi-Level Comparison
====================================

Main demos:

* `_demos/general/demo_multi_term_vs_multi_level_divergence.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_multi_term_vs_multi_level_divergence.py>`_
* `_demos/general/demo_multi_term_vs_multi_level_S0.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_multi_term_vs_multi_level_S0.py>`_
* `_demos/general/demo_multi_term_vs_multi_level_S0_lte_nlte.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_multi_term_vs_multi_level_S0_lte_nlte.py>`_

The multi-term and multi-level descriptions can be applied to the same atmosphere and radiation field. This is the main diagnostic use of SolRaT: the user can change the atomic description without changing the surrounding transfer problem.

The divergence demo shows two effects. In strong fields, the multi-term atom redistributes magnetic-component strengths through Paschen-Back mixing, while the multi-level atom remains in the linear-Zeeman description. In weak-field non-LTE scattering, multi-term and multi-level calculations can differ through the inter-:math:`J` coherences retained by the multi-term description.

The ``S=0`` demos provide a control case. For singlet terms, each term has only one fine-structure level, so the multi-term and multi-level descriptions reduce to the same problem and should agree numerically.
