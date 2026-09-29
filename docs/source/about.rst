About SolRaT
============

SolRaT is a forward-modeling code for polarized spectral-line synthesis in the density-matrix formalism. It provides multi-term and multi-level atomic descriptions behind the same atmosphere and transfer interfaces, so the user can change the atomic description while keeping the atmosphere, geometry, incident radiation, and formal transfer setup fixed.

Features
--------

* Density-matrix statistical tensors :math:`\rho^K_Q` for atomic polarization.
* Multi-term atoms with inter-:math:`J` coherences and linear-Zeeman through incomplete and complete Paschen-Back splitting.
* Multi-level atoms with empirical level energies and linear-Zeeman splitting.
* LTE statistical-equilibrium variants that reuse the same RTE implementation.
* Constant-property slabs, multi-slab sequences, prescribed-:math:`J^K_Q` stratified atmospheres, and self-consistent stratified non-LTE atmospheres.
* A summation engine that keeps SEE rates and RTE coefficients close to their mathematical form while keeping expensive reusable factors cached.

Reference Article
-----------------

The SolRaT article contains the physical conventions, reference frames, sign conventions, equations, validation tests, and examples used by the code:

    Yakovkin, I. I. 2026, *SolRaT: polarized spectral line modeling with multi-term and multi-level atoms*, `arXiv:2609.32850 <https://arxiv.org/abs/2609.32850>`_.

Developers
----------

The package is developed by Ivan I. Yakovkin. Contributions and issue reports should go through the `GitHub repository <https://github.com/yakovkinii/SolRaT/>`_.
