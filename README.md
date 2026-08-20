# SolRaT 

[![Documentation](https://img.shields.io/badge/read-TheDocs-eee?logoColor=black)](https://solrat.readthedocs.io/latest/)
[![Homepage](https://img.shields.io/badge/homepage-solrat-000000?logoColor=white)](https://www.yakovkinii.com/solrat/)
![License](https://img.shields.io/badge/license-MIT-00ff00)
[![PyPi Version](https://img.shields.io/pypi/v/solrat)](https://pypi.org/project/solrat)
![Language](https://img.shields.io/badge/language-Python-3776AB?logoColor=white)
![Supported Platforms](https://img.shields.io/badge/platform-any-ffffff?logoColor=black)
[![Coverage Status](https://coveralls.io/repos/github/yakovkinii/SolRaT/badge.svg?branch=master)](https://coveralls.io/github/yakovkinii/SolRaT?branch=master)

SolRaT (Solar Radiative Transfer) is a forward-modeling code for the polarized, non-LTE
transfer of spectral-line radiation in magnetized stellar atmospheres. It is built on the
density-matrix formalism of [[LL04](#References)] and written so that each statistical-equilibrium
and radiative-transfer expression reads close to the equation it implements. The aim is a
model that is transparent enough to inspect and verify, and flexible enough to adapt to a
specific line or context rather than used as a black box.

#### Physical model
- **Density-matrix formalism** in the irreducible spherical statistical tensors $\rho^K_Q$,
with atomic level polarization fully included [[LL04](#References)].
- **Interchangeable atomic models** in a single pipeline: multi-term, multi-level, and a
semi-LTE multi-term model, selectable without rewriting the surrounding code.
- **Magnetic fields of arbitrary strength**: Zeeman, Hanle, and the Paschen-Back regime by
exact diagonalization of the atomic Hamiltonian (multi-term atom; Zeeman and Hanle for the
multi-level atom).
- **Radiation field** $J^K_Q$ either prescribed (LTE Planck, or the anisotropic $\{n, w\}$
parametrization of [[ATL08](#References)] for coronal/chromospheric lines) or solved
self-consistently for the non-LTE scattering problem [[TB99](#References)].

#### Atmospheres and synthesis
- **Constant-property slabs**, optionally stacked into a multi-slab stratification under
anisotropic illumination.
- **Height-stratified atmosphere** in which temperature, absorber number density, the
magnetic-field vector, microturbulence, Voigt damping, and the vector macroscopic velocity
vary continuously with geometric height. The scattering $J^K_Q$ is solved self-consistently
by $\Lambda$-iteration on a depth grid, with the Stokes transfer solved by the DELO method.
- Emergent Stokes profiles for a chosen line of sight at arbitrary spectral resolution.

#### Design
SolRaT is organized in three layers:
- a **public API** to run the built-in models;
- a **modeling API** to extend a model or build a new one by analogy with the shipped ones;
- the **SolRaT engine**, a vectorized meta-language in which the angular algebra and rate
expressions are written close to their mathematical form, with the bookkeeping and
optimization handled underneath.

Pre-configured lines: He I D3, Mn I 5432.5 &Aring;, Ni I 5435.9 &Aring;, Fe I 5434.523 &Aring;.

#### Scope and limitations
SolRaT is a forward model. Its non-LTE solution is collisionless (pure scattering) by default,
so scattering-polarization amplitudes are then upper limits; an optional
parametrized-collision extension (for both the multi-level and the multi-term atom) adds inelastic
(transfer) and elastic (depolarizing) rates that bridge the scattering limit to LTE. Line formation assumes complete
frequency redistribution (CRD). Physical
collisional rates from cross-sections, partial frequency redistribution, and 3D geometry are out
of scope for the current version.

#### Installation
Install SolRaT directly from PyPi by running ```pip install solrat```.

#### Documentation
Detailed documentation is available at [https://solrat.readthedocs.io/](https://solrat.readthedocs.io/latest/). 
A quick-start example is available at [https://solrat.readthedocs.io/latest/quickstart.html](https://solrat.readthedocs.io/latest/quickstart.html).
Additional demos and validation against [[LL04](#References)] and [[HAZEL2](#References)] are available in [demos](https://github.com/yakovkinii/SolRaT/tree/master/_demos). 

#### Citing
A journal article is in preparation. In the meantime, if SolRaT has found use in your research, please cite it as 
```
Yakovkin I. I. SolRaT (2023) [computer software]. Retrieved from https://www.yakovkinii.com/solrat/
```

#### References
[LL04] Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)

[ATL08] Asensio Ramos, A., Trujillo Bueno, J., & Landi Degl’Innocenti, E. (2008). Advanced Forward Modeling and Inversion of Stokes Profiles Resulting from the Joint Action of the Hanle and Zeeman Effects. The Astrophysical Journal, 683(1), 542–565.

[TB99] Trujillo Bueno, J., & Manso Sainz, R. (1999). Iterative Methods for the Non-LTE Transfer of Polarized Radiation: Resonance Line Polarization in One-dimensional Atmospheres. The Astrophysical Journal, 516(1), 436–450.

[HAZEL2] [Link](https://github.com/aasensio/hazel2)

<h4>Keywords:</h4>
Non-LTE, Stokes Profiles, Synthesis, Paschen-Back, Hanle, Zeeman, 
Magnetic Fields, Sun, Solar Atmosphere, Radiative Transfer, Spectral Line Polarization, 
Spectral Lines, Multi-Term Atom Model, Multi-Level Atom Model, Atomic Polarization. 

Copyright (2023) Ivan I. Yakovkin
