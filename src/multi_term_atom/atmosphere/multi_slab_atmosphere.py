import numpy as np
from typing import List, Union
from .constant_property_slab import ConstantPropertySlab
from .linear_property_slab import LinearPropertySlab
from src.multi_term_atom.object.stokes import Stokes


class MultiSlabAtmosphere:
    """
    Container that combines multiple slabs to create a stratified atmosphere.
    Supports both sequential propagation and filling factor mixing.
    """

    def __init__(self):
        self.slabs = []
        self.filling_factors = []

    def add_slab_sequential(self, slab: Union[ConstantPropertySlab, LinearPropertySlab]):
        """Add slab for sequential propagation (one after another)"""
        self.slabs.append(slab)
        self.filling_factors.append(1.0)  # Full coverage for sequential

    def add_slab_filling_factor(self, slab: Union[ConstantPropertySlab, LinearPropertySlab],
                               filling_factor: float):
        """Add slab with specified filling factor for mixed atmosphere"""
        assert 0 <= filling_factor <= 1, "Filling factor must be between 0 and 1"
        self.slabs.append(slab)
        self.filling_factors.append(filling_factor)

    def forward_sequential(self, initial_stokes: Union[Stokes, float, None] = None) -> Stokes:
        """
        Propagate radiation through slabs sequentially (one after another).
        Each slab uses the output of the previous slab as input.
        """
        if not self.slabs:
            raise ValueError("No slabs added to atmosphere")

        # Set initial conditions for first slab
        if initial_stokes is not None:
            self.slabs[0].initial_stokes = initial_stokes

        # Propagate through each slab sequentially
        current_stokes = self.slabs[0].forward()

        for i in range(1, len(self.slabs)):
            # Use output of previous slab as input to next slab
            self.slabs[i].initial_stokes = current_stokes
            current_stokes = self.slabs[i].forward()

        return current_stokes

    def forward_filling_factors(self, initial_stokes: Union[Stokes, float, None] = None) -> Stokes:
        """
        Combine slabs using filling factors (weighted average of emergent intensities).
        Each slab sees the same initial boundary condition.
        """
        if not self.slabs:
            raise ValueError("No slabs added to atmosphere")

        # Normalize filling factors
        total_ff = sum(self.filling_factors)
        if total_ff == 0:
            raise ValueError("Total filling factors cannot be zero")
        normalized_ff = [ff / total_ff for ff in self.filling_factors]

        # Set same initial conditions for all slabs
        for slab in self.slabs:
            if initial_stokes is not None:
                slab.initial_stokes = initial_stokes

        # Compute weighted average of emergent Stokes vectors
        stokes_results = [slab.forward() for slab in self.slabs]

        # Initialize combined result with first slab
        combined_nu = stokes_results[0].nu
        combined_I = normalized_ff[0] * stokes_results[0].I
        combined_Q = normalized_ff[0] * stokes_results[0].Q
        combined_U = normalized_ff[0] * stokes_results[0].U
        combined_V = normalized_ff[0] * stokes_results[0].V

        # Add contributions from other slabs
        for i in range(1, len(stokes_results)):
            combined_I += normalized_ff[i] * stokes_results[i].I
            combined_Q += normalized_ff[i] * stokes_results[i].Q
            combined_U += normalized_ff[i] * stokes_results[i].U
            combined_V += normalized_ff[i] * stokes_results[i].V

        return Stokes(
            nu=combined_nu,
            I=combined_I,
            Q=combined_Q,
            U=combined_U,
            V=combined_V,
        )

    def get_total_optical_depth(self) -> float:
        """Calculate total optical depth for sequential configuration"""
        return sum(slab.tau for slab in self.slabs)

    def get_effective_filling_factor(self) -> float:
        """Get effective filling factor (useful for diagnostics)"""
        return sum(self.filling_factors)


# Example usage functions
def create_two_component_atmosphere(context, radiation_tensor,
                                  tau1=1.0, B1=1000, tau2=2.0, B2=5000,
                                  filling_factor1=0.7, filling_factor2=0.3):
    """Create a simple two-component atmosphere with different magnetic fields"""
    atmosphere = MultiSlabAtmosphere()

    # Component 1: Quiet region
    slab1 = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=tau1,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=B1,
        delta_v_thermal_cm_sm1=4000,
    )
    atmosphere.add_slab_filling_factor(slab1, filling_factor1)

    # Component 2: Active region
    slab2 = ConstantPropertySlab(
        multi_term_atom_context=context,
        radiation_tensor=radiation_tensor,
        tau=tau2,
        chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
        magnetic_field_gauss=B2,
        delta_v_thermal_cm_sm1=6000,
    )
    atmosphere.add_slab_filling_factor(slab2, filling_factor2)

    return atmosphere


def create_stratified_atmosphere(context, radiation_tensor, n_layers=5):
    """Create a stratified atmosphere with gradually increasing magnetic field"""
    atmosphere = MultiSlabAtmosphere()

    base_tau = 0.5
    base_B = 500

    for i in range(n_layers):
        # Exponentially increasing magnetic field with height (deeper = stronger)
        B_field = base_B * (1.5 ** i)
        tau_layer = base_tau

        slab = ConstantPropertySlab(
            multi_term_atom_context=context,
            radiation_tensor=radiation_tensor,
            tau=tau_layer,
            chi=0, theta=0, gamma=0, chi_B=0, theta_B=0,
            magnetic_field_gauss=B_field,
            delta_v_thermal_cm_sm1=4000 + i * 500,  # Also vary thermal velocity
        )
        atmosphere.add_slab_sequential(slab)

    return atmosphere
