from src.multi_term_atom.atmosphere.constant_property_slab import (
    ConstantPropertySlabAtmosphere,
)
from src.multi_term_atom.object.stokes import Stokes


class MultiSlabAtmosphere:
    def __init__(self, *slabs: ConstantPropertySlabAtmosphere):
        """
        Container that consecutively combines multiple slabs to create a stratified atmosphere.
        """
        self.slabs = slabs

    def forward(self, initial_stokes: Stokes) -> Stokes:
        """
        Propagate radiation through slabs sequentially (one after another).
        Each slab uses the output of the previous slab as input.

        :param initial_stokes:  Initial Stokes vector that is entering the slab.
        """

        current_stokes = self.slabs[0].forward(initial_stokes=initial_stokes)

        for i in range(1, len(self.slabs)):
            current_stokes = self.slabs[i].forward(initial_stokes=current_stokes)

        return current_stokes
