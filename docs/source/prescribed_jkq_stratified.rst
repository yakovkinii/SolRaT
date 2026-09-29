Prescribed-JKQ Stratified Atmosphere
====================================

Main demo:

`_demos/general/demo_unno_rachkovsky_ME.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_unno_rachkovsky_ME.py>`_

:class:`~solrat.atom_model.shared.common_api.stratified_nlte_atmosphere.PrescribedRadiationStratifiedAtmosphere` keeps the formal stratified transfer machinery but does not iterate the radiation field. Instead, the user supplies a prescribed radiation tensor :math:`J^K_Q` on the depth grid, or a rule from which the code constructs it.

This is useful when a height-dependent atmosphere is needed but a full self-consistent scattering calculation is not. Temperature, number density, magnetic field, velocity, damping, and continuum opacity can vary with height, while the radiation tensor is imposed externally.

The Unno-Rachkovsky demo uses this interface as a validation limit: it constructs a stratified transfer problem whose analytic Milne-Eddington solution is known, then compares the numerical formal solution against that reference.
