"""
TODO
TODO  This file needs improved documentation.
TODO
"""

from typing import Callable, Union

import numpy as np
from scipy.linalg import expm

from src.common.functions import lambda_cm_to_frequency_hz
from src.multi_term_atom.object.multi_term_atom_context import MultiTermAtomContext
from src.multi_term_atom.object.radiation_tensor import RadiationTensor
from src.multi_term_atom.object.stokes import Stokes

from .constant_property_slab import ConstantPropertySlab


class LinearPropertySlab(ConstantPropertySlab):
    """
    Slab with linearly varying properties between top and bottom.
    Uses multiple sub-slabs with constant properties to approximate the gradient.
    """

    def __init__(
        self,
        multi_term_atom_context: MultiTermAtomContext,
        radiation_tensor: RadiationTensor,
        tau_total: float,
        # Linear variation parameters
        magnetic_field_gauss_top: float,
        magnetic_field_gauss_bottom: float,
        delta_v_thermal_cm_sm1_top: float,
        delta_v_thermal_cm_sm1_bottom: float,
        # Geometry parameters
        chi: float,
        theta: float,
        gamma: float,
        chi_B: float,
        theta_B: float,
        # Optional parameters
        macroscopic_velocity_cm_sm1: float = 0,
        voigt_a: float = 0,
        n_sub_slabs: int = 10,
        initial_stokes: Union[Stokes, float, None] = None,
    ):
        self.n_sub_slabs = n_sub_slabs
        self.tau_total = tau_total
        self.dtau = tau_total / n_sub_slabs

        # Store gradient parameters
        self.B_top = magnetic_field_gauss_top
        self.B_bottom = magnetic_field_gauss_bottom
        self.v_th_top = delta_v_thermal_cm_sm1_top
        self.v_th_bottom = delta_v_thermal_cm_sm1_bottom

        # Store geometry
        self.chi = chi
        self.theta = theta
        self.gamma = gamma
        self.chi_B = chi_B
        self.theta_B = theta_B
        self.macroscopic_velocity = macroscopic_velocity_cm_sm1
        self.voigt_a = voigt_a

        # Store context and other parameters
        self.context = multi_term_atom_context
        self.radiation_tensor = radiation_tensor
        self.lambda_A = multi_term_atom_context.lambda_A
        self.reference_lambda_A = multi_term_atom_context.reference_lambda_A

        # Handle initial conditions
        if isinstance(initial_stokes, Stokes):
            self.initial_stokes_obj = initial_stokes
        elif isinstance(initial_stokes, (int, float)):
            self.initial_stokes_value = initial_stokes
        else:
            self.initial_stokes_value = 1.0

    def get_properties_at_depth(self, tau_fraction: float) -> tuple:
        """Get linearly interpolated properties at given optical depth fraction (0=top, 1=bottom)"""
        B_field = self.B_top + tau_fraction * (self.B_bottom - self.B_top)
        v_thermal = self.v_th_top + tau_fraction * (self.v_th_bottom - self.v_th_top)
        return B_field, v_thermal

    def forward(self) -> Stokes:
        """Solve radiative transfer through linearly varying medium using sub-slab method"""
        # Initialize with boundary condition
        if hasattr(self, "initial_stokes_obj"):
            current_stokes = self.initial_stokes_obj
        else:
            nu = lambda_cm_to_frequency_hz(self.lambda_A * 1e-8)
            current_stokes = Stokes(
                nu=nu,
                I=np.full_like(nu, self.initial_stokes_value, dtype=np.float64),
                Q=np.zeros_like(nu, dtype=np.float64),
                U=np.zeros_like(nu, dtype=np.float64),
                V=np.zeros_like(nu, dtype=np.float64),
            )

        # Propagate through each sub-slab
        for i in range(self.n_sub_slabs):
            tau_fraction = (i + 0.5) / self.n_sub_slabs  # Mid-point of sub-slab
            B_field, v_thermal = self.get_properties_at_depth(tau_fraction)

            # Create constant property sub-slab
            sub_slab = ConstantPropertySlab(
                multi_term_atom_context=self.context,
                radiation_tensor=self.radiation_tensor,
                tau=self.dtau,
                chi=self.chi,
                theta=self.theta,
                gamma=self.gamma,
                magnetic_field_gauss=B_field,
                chi_B=self.chi_B,
                theta_B=self.theta_B,
                delta_v_thermal_cm_sm1=v_thermal,
                macroscopic_velocity_cm_sm1=self.macroscopic_velocity,
                voigt_a=self.voigt_a,
                initial_stokes=current_stokes,
            )

            # Solve this sub-slab and update Stokes vector
            current_stokes = sub_slab.forward()

        return current_stokes
