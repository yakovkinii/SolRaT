Modifying RTE
=============

The multi-level RTE implementation is the most compact starting point:

`solrat/atom_model/multi_level_atom_model/radiative_transfer_equations.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_level_atom_model/radiative_transfer_equations.py>`_

The central class is :class:`~solrat.atom_model.multi_level_atom_model.radiative_transfer_equations.MultiLevelAtomRTE`. The main coefficient-building methods are:

* ``calculate_eta_rho_a`` for absorption and anomalous dispersion from lower-level tensors.
* ``calculate_eta_rho_s`` for stimulated-emission opacity and anomalous dispersion from upper-level tensors.
* ``calculate_epsilon`` for emissivity from stimulated emission.
* ``calculate_all_coefficients`` for packaging the complete set of RTE coefficients.

The multi-term counterpart is `solrat/atom_model/multi_term_atom_model/radiative_transfer_equations.py <https://github.com/yakovkinii/SolRaT/blob/master/solrat/atom_model/multi_term_atom_model/radiative_transfer_equations.py>`_. It adds Paschen-Back eigenvectors and field-shifted component frequencies.

Summation Order
---------------

RTE coefficients are built as constrained sums with the SolRaT engine. When adding a new factor or a new summation, pay attention to the order of indexes passed to ``reduce()`` or ``reduce_partially()``. Different reduction orders can give the same mathematics but different intermediate table sizes. If a new term becomes slow, try reducing the largest or most independent index groups earlier, and compare the resulting performance.

RTE Caching
-----------

There are two relevant cache levels.

First, atom-level frame construction is stored on the RTE object. This is valid as long as the atom and frequency grid represented by that RTE instance do not change.

Second, the operator cache stores compiled coefficient operators for a fixed atmosphere and geometry while only the density tensors vary. This is useful inside self-consistent NLTE iterations and is controlled by ``use_operator_cache`` and ``clear_operator_cache`` on the RTE object. If a new RTE factor depends on atmosphere parameters, geometry, or frequency, make sure it is included before the operator is cached. If it depends on the density tensor, it should remain outside the operator cache and enter through the final multiplication by ``rho``.

For prescribed-radiation and slab atmospheres, operator caching is kept disabled. For self-consistent stratified atmospheres, the atmosphere enables the operator cache during the iteration and clears it when the forward call exits.
