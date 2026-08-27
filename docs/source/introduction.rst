SolRaT: Introduction
====================

SolRaT (Solar Radiative Transfer) is a forward-modeling code for the polarized, non-LTE transfer of spectral-line radiation in magnetized stellar atmospheres. It solves the statistical-equilibrium and radiative-transfer equations in the density-matrix formalism of Landi Degl'Innocenti & Landolfi (2004, LL04), for magnetic fields of arbitrary strength, from the Zeeman through the Hanle and Paschen-Back regimes.

.. image:: https://www.yakovkinii.com/solrat/media/solrat7.png
   :width: 600
   :alt: Model of He I D3 emission in a limb event under different magnetic fields
   :align: center

SolRaT is written so that each rate and transfer expression reads close to the equation it implements. The goal is a model transparent enough to inspect and verify against analytic results and established codes, and flexible enough to adapt to a particular line or context rather than be used as a black box. The figure above shows a sample output: He I D3 emission modeled under varying magnetic field strengths.

Physical model
--------------
*   **Density-matrix formalism** in the irreducible spherical statistical tensors :math:`\rho^K_Q`, with atomic level polarization fully included (LL04).
*   **Interchangeable atomic models** in a single pipeline: multi-term, multi-level, and LTE variants of both descriptions, selectable without rewriting the surrounding code.
*   **Magnetic fields of arbitrary strength**: Zeeman, Hanle, and the Paschen-Back regime by exact diagonalization of the atomic Hamiltonian (multi-term atom; Zeeman and Hanle for the multi-level atom).
*   **Radiation field** :math:`J^K_Q` either prescribed (LTE Planck, or Allen/ATL08-style anisotropic :math:`\{n, w\}` values for coronal/chromospheric lines) or solved self-consistently for the non-LTE scattering problem (TM99): collisionless by default, with optional parametrized collisional rates in the statistical equilibrium of both the multi-level and the multi-term atom (inelastic transfer with Einstein-Milne detailed balance, and elastic depolarization) that bridge the scattering limit to LTE.

Atmospheres and synthesis
-------------------------
*   **Constant-property slabs**, optionally stacked into a multi-slab stratification under anisotropic illumination (ATL08, HAZEL2).
*   **Height-stratified atmosphere** in which temperature, absorber number density, the magnetic-field vector, microturbulence, Voigt damping, and the vector macroscopic velocity vary continuously with geometric height; the scattering :math:`J^K_Q` is solved self-consistently by :math:`\Lambda`-iteration on a depth grid, with the Stokes transfer solved by the DELO method.
*   Emergent Stokes profiles for a chosen line of sight at arbitrary spectral resolution.

Design
------
SolRaT is organized in three layers:

*   **Public API** runs the built-in models.
*   **Modeling API** extends an existing model or builds a new one by analogy with the shipped ones.
*   **SolRaT engine** is a vectorized meta-language in which the angular algebra and rate expressions are written close to their mathematical form; the bookkeeping and optimization are handled underneath, so the user can focus on the physics rather than on code optimization.

Scope and limitations
---------------------
SolRaT is a forward model. Its non-LTE solution is collisionless (pure scattering) by default, so scattering-polarization amplitudes are then upper limits; an optional parametrized-collision extension (for both the multi-level and the multi-term atom) adds inelastic (transfer) and elastic (depolarizing) rates that bridge the scattering limit to LTE via detailed balance. Line formation assumes complete frequency redistribution (CRD). Physical collisional rates from cross-sections, partial frequency redistribution, and 3D geometry are out of scope for the current version.

References
----------
[LL04] Landi Degl’Innocenti, E., & Landolfi, M. 2004, Polarization in Spectral Lines (Dordrecht: Kluwer)

[ATL08] Asensio Ramos, A., Trujillo Bueno, J., & Landi Degl’Innocenti, E. (2008). Advanced Forward Modeling and Inversion of Stokes Profiles Resulting from the Joint Action of the Hanle and Zeeman Effects. The Astrophysical Journal, 683(1), 542–565.

[TM99] Trujillo Bueno, J., & Manso Sainz, R. (1999). Iterative Methods for the Non-LTE Transfer of Polarized Radiation. The Astrophysical Journal, 516(1), 436–450.

[HAZEL2] https://github.com/aasensio/hazel2

How to Cite
-----------
A journal article is in preparation. In the meantime, if SolRaT contributes to your research, please cite it as:

    Yakovkin I. I. SolRaT (2023) [computer software]. Retrieved from https://www.yakovkinii.com/solrat/

Installation
------------
Install the latest release from PyPI:

.. code-block:: bash

   pip install solrat

For the development version, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/yakovkinii/SolRaT.git
   cd SolRaT
   pip install -e .

Next
----
Check out the :doc:`quickstart` guide for basic usage examples.
