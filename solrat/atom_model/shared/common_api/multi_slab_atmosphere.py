from typing import Protocol

from solrat.atom_model.shared.object.stokes import Stokes
from solrat.engine.functions.decorators import log_method


class ForwardAtmosphere(Protocol):
    def forward(self, initial_stokes: Stokes) -> Stokes:
        ...


class MultiSlabAtmosphere:
    r"""
    Container that propagates Stokes vectors through several atmosphere pieces in sequence.

    Each piece only needs to expose ``forward(initial_stokes)``. This allows, for example,
    constant-property slabs with different atomic descriptions, or a mixture of slab and
    stratified atmosphere pieces.
    """

    def __init__(self, *slabs: ForwardAtmosphere):
        if not slabs:
            raise ValueError("MultiSlabAtmosphere requires at least one slab.")  # pragma: no cover
        self.slabs = slabs

    @log_method
    def forward(self, initial_stokes: Stokes) -> Stokes:
        r"""
        Propagate radiation through slabs sequentially (one after another).
        Each slab uses the output of the previous slab as input.

        :param initial_stokes:  Initial Stokes vector that is entering the slab.
        """

        current_stokes = self.slabs[0].forward(initial_stokes=initial_stokes)

        for i in range(1, len(self.slabs)):
            current_stokes = self.slabs[i].forward(initial_stokes=current_stokes)

        return current_stokes
