import logging

import numpy as np
from yatools import logging_config

from src.core.engine.functions.looping import FROMTO, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.core.physics.functions import get_planck_BP
from src.core.physics.rotations import T_K_Q, T_from_t, WignerD

# from src.two_term_atom.object.atmosphere_parameters import AtmosphereParameters
from src.two_term_atom.object.radiation_tensor import RadiationTensor
from src.two_term_atom.terms_levels_transitions.term_registry import TermRegistry
from src.two_term_atom.terms_levels_transitions.transition_registry import TransitionRegistry


# TODO
def main():
    logging_config.init(logging.INFO)

    term_registry = TermRegistry()
    term_registry.register_term(
        beta="1s",
        L=0,
        S=0.5,
        J=0.5,
        energy_cmm1=200_000,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0.5,
        J=0.5,
        energy_cmm1=220_000,
    )
    term_registry.register_term(
        beta="2p",
        L=1,
        S=0.5,
        J=1.5,
        energy_cmm1=220_001,
    )
    term_registry.validate()

    # nu = np.arange(5.995e14, 5.997e14, 1e8)  # Hz
    nu = np.array([5.996e14])

    transition_registry = TransitionRegistry()
    transition_registry.register_transition_from_a_ul(
        level_upper=term_registry.get_level(beta="2p", L=1, S=0.5),
        level_lower=term_registry.get_level(beta="1s", L=0, S=0.5),
        einstein_a_ul_sm1=0.7e8,
    )

    # atmosphere_parameters = AtmosphereParameters(magnetic_field_gauss=0, delta_v_thermal_cm_sm1=5_000_00)
    radiation_tensor = RadiationTensor(transition_registry=transition_registry)
    I0 = get_planck_BP(nu_sm1=nu, T_K=5000)
    radiation_tensor.fill_NLTE_near_isotropic(I0)

    # Define Omega
    # chi is angle between Bxy and X
    # theta is angle between Bz and Z
    # alpha is angle between \perp B and Xnew, i.e. Stokes Q/U reference.
    # gamma=0 -> vector Omega a is looking down (Fig. 5.14)
    # euler angles are alpha=chi, beta=theta, gamma=gamma
    chi = np.pi / 5
    theta = 0 * np.pi / 3
    gamma = 0 * np.pi / 4

    D = WignerD(alpha=chi, beta=theta, gamma=gamma, K_max=2)
    # Dm1 = WignerD(alpha=-gamma, beta=-theta, gamma=-chi, K_max=2)

    # # rot1j = rotate_J(J=radiation_tensor, D=D)
    # rotated_J = rotate_J(rotate_J(J=radiation_tensor, D=D), D=Dm1)
    # a = 2

    for K, Q, stokes_component_index in nested_loops(
        K=FROMTO(0, 2), Q=PROJECTION("K"), stokes_component_index=FROMTO(0, 3)
    ):
        T1 = T_from_t(K=K, Q=Q, stokes_component_index=stokes_component_index, D=D)
        T2 = T_K_Q(K=K, Q=Q, stokes_component_index=stokes_component_index, chi=chi, theta=theta, gamma=gamma)
        print(f"{K=} {Q=} {stokes_component_index=}")
        print(f"T1 = {T1}")
        print(f"T2 = {T2}")


if __name__ == "__main__":
    main()
