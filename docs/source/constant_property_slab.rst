Constant-Property Slab Synthesis
================================

Main demo:

`_demos/multi_term_atom/demo_constant_property_slab_HeI_D3.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/multi_term_atom/demo_constant_property_slab_HeI_D3.py>`_

A constant-property slab represents a homogeneous layer with one set of thermodynamic, magnetic, velocity, and line-profile parameters. The user supplies the optical thickness in the line and continuum, the incident Stokes vector, the observation geometry, the radiation tensor used by the SEE, and the atmosphere parameters used by the RTE.

This atmosphere is useful when the goal is to isolate the response of one spectral line to a controlled magnetic field, radiation tensor, or atomic description. It is also the simplest setting for comparing multi-term and multi-level descriptions under identical conditions.

In the demo, :class:`~solrat.atom_model.shared.common_api.constant_property_slab.ConstantPropertySlabAtmosphere` is wrapped in :class:`~solrat.atom_model.shared.common_api.multi_slab_atmosphere.MultiSlabAtmosphere` with only one slab. This keeps the same calling pattern that can later be extended to several slabs without changing the synthesis interface.
