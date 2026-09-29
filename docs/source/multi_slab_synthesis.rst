Multi-Slab Synthesis
====================

Representative demos:

* `_demos/start_here/demo_basic_stokes_profile_synthesis.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/start_here/demo_basic_stokes_profile_synthesis.py>`_
* `_demos/multi_term_atom_lte/demo_constant_property_slab_MnI_FeI_NiI.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/multi_term_atom_lte/demo_constant_property_slab_MnI_FeI_NiI.py>`_

:class:`~solrat.atom_model.shared.common_api.multi_slab_atmosphere.MultiSlabAtmosphere` propagates a Stokes vector through a sequence of atmosphere pieces. Each piece only needs to expose ``forward(initial_stokes)``, so the sequence can contain constant-property slabs and, in principle, other compatible atmosphere objects.

The idea is to build a piecewise model of the line of sight: the emergent Stokes vector from one slab becomes the incident Stokes vector for the next. This is appropriate when the physical conditions are naturally separated into a small number of homogeneous components, for example several emitting or absorbing structures along the ray.

Each slab can use its own atmosphere parameters and radiation tensor. This makes multi-slab synthesis a practical way to combine different components without introducing a continuous height grid.
