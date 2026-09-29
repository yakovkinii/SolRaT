Self-Consistent NLTE Atmosphere
===============================

Representative demos:

* `_demos/general/demo_nlte_TM99_resonance_polarization.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_nlte_TM99_resonance_polarization.py>`_
* `_demos/general/demo_nlte_TM99_resonance_polarization_mu01.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_nlte_TM99_resonance_polarization_mu01.py>`_
* `_demos/general/demo_nlte_thermalization_sqrt_epsilon.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_nlte_thermalization_sqrt_epsilon.py>`_
* `_demos/general/demo_nlte_stratified_atmosphere.py <https://github.com/yakovkinii/SolRaT/blob/master/_demos/general/demo_nlte_stratified_atmosphere.py>`_

:class:`~solrat.atom_model.shared.common_api.stratified_nlte_atmosphere.NLTEStratifiedAtmosphere` solves the SEE and RTE on a height grid. The radiation tensor is initialized, the SEE is solved at each depth, the Stokes vector is formally propagated, and the radiation tensor is updated by angular quadrature. This loop continues until the convergence criterion is met or the maximum iteration count is reached.

The atmosphere supports height-dependent temperature, number density, magnetic field, velocity, Voigt damping, and continuum opacity. It also supports warm starts through :class:`~solrat.atom_model.shared.common_api.nlte_state.NLTEState`. Several benchmark demos load saved states from `_demos/general/state/ <https://github.com/yakovkinii/SolRaT/tree/master/_demos/general/state>`_; if one of these demos is copied elsewhere, copy the corresponding ``.npz`` state file too. Without it, the calculation starts from a cold LTE guess and may take longer to converge.

The TM99 demos reproduce resonance-line polarization benchmarks. The sqrt-epsilon demo checks thermalization behavior against a classic unpolarized benchmark.
