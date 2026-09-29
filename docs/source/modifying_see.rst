Modifying SEE
=============

The multi-level SEE implementation is the simplest place to start:

`solrat/atom_model/multi_level_atom_model/statistical_equilibrium_equations.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_level_atom_model/statistical_equilibrium_equations.py>`_

The central class is :class:`~solrat.atom_model.multi_level_atom_model.statistical_equilibrium_equations.MultiLevelAtomSEE`. Its ``fill_all_equations`` method assembles the linear SEE system by calling the rate-building methods:

* ``add_coherence_decay`` mirrors the magnetic precession term.
* ``add_absorption`` mirrors the absorption transfer rate.
* ``add_emission_e`` and ``add_emission_s`` mirror spontaneous and stimulated emission transfer.
* ``add_relaxation_e``, ``add_relaxation_a``, and ``add_relaxation_s`` mirror the relaxation terms.
* ``add_collisions`` adds the parametrized collisional rates.

The method docstrings identify the corresponding LL04 equations. The multi-term counterpart is in `solrat/atom_model/multi_term_atom_model/statistical_equilibrium_equations.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_term_atom_model/statistical_equilibrium_equations.py>`_ and follows the same organization with term-level and inter-:math:`J` coherences.

How to Modify
-------------

For a local experiment, modify the existing class directly. For a reusable model, copy the relevant model package or class, give the modified description a new name, and register it in :mod:`solrat.atom_model.model_registry`.

SEE Caching
-----------

SEE frames are cached as instance attributes such as ``absorption_frame`` or ``relaxation_s_frame``. These are atom-level caches: they depend on the selected atomic configuration and angular-momentum algebra, not on one particular atmospheric depth. If a new SEE term adds a new fixed summation structure, follow the existing pattern by storing its prepared frame on the SEE instance. If the atomic configuration changes, construct a new SEE object rather than trying to reuse the old cached frames.
