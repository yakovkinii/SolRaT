Adding More Spectral Lines
==========================

The best multi-term example is:

`solrat/atom_model/multi_term_atom_model/data/HeI.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_term_atom_model/data/HeI.py>`_

The direct multi-level analog is:

`solrat/atom_model/multi_level_atom_model/data/HeI.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_level_atom_model/data/HeI.py>`_

To add a line, create a level registry, register every relevant fine-structure level, create a transition registry, and register the radiative transitions. Then expose the configuration through :mod:`solrat.atom_model.model_registry`.

Atomic Data
-----------

Level energies, wavelengths, Landé factors, and Einstein :math:`A` values can be taken from NIST when available. For a multi-level atom, a transition connects two fine-structure levels, so the NIST component value is used directly for that component.

For a multi-term atom, a transition connects two terms and SolRaT expects one term-to-term Einstein :math:`A` value. NIST usually gives component values :math:`A(\beta_u L_u S J_u \to \beta_l L_l S J_l)`. The guide in `HeI.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_term_atom_model/data/HeI.py>`_ gives the rule: for each upper :math:`J_u`, sum over the lower :math:`J_l` components, then average those sums over the distinct upper :math:`J_u` values. Do not use the raw sum over all components unless the upper term has only one :math:`J_u`.

After adding a new configuration, add a small demo or test that exercises the new atom over the intended wavelength range.
