from typing import Union

import numpy as np
import pandas as pd

from src.engine.functions.general import delta
from src.engine.functions.looping import FROMTO, PROJECTION
from src.engine.generators.nested_loops import nested_loops
from src.engine.generators.summate import summate
from src.engine.objects.container import Container
from src.common.constants import c_cm_sm1, h_erg_s, sqrt2
from src.common.functions import frequency_hz_to_lambda_A, get_planck_BP
from src.common.rotations import WignerD
from src.multi_term_atom.terms_levels_transitions.transition_registry import Transition, TransitionRegistry


class RadiationTensor(Container):
    def __init__(self, transition_registry: TransitionRegistry):
        """
        Radiation tensor J^K_Q (nu_ul).
        Here we assume that transitions are spread apart in frequency, so that we can assign a bijection of
        transition <-> nu_ul, and store J for each transition instead for clarity.
        I believe K is always <=2 by construction (see eg. 5.157) for electric-dipole transitions due to T tensor.
        """
        super().__init__()
        self.transition_registry = transition_registry
        self.df: Union[pd.DataFrame, None] = None

    def fill_planck(self, T_K: float):
        """
        Flat-spectrum approximation, i.e. J needs to be defined for each transition, not for each frequency.
        """
        assert self.df is None, "Cannot modify radiation tensor after it has been constructed"
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            planck = get_planck_BP(nu_sm1=nu_ul, T_K=T_K)
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = planck * delta(K, 0) * delta(Q, 0)
        self.construct_df()
        return self

    @staticmethod
    def n_fit(lambda_A):
        """
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        """
        assert lambda_A >= 2750, "n_fit is only valid for lambda_A >= 2750"
        assert lambda_A <= 12000, "n_fit is not tested for lambda_A >= 12000"
        return 3e-10 * (lambda_A - 2500) ** 2.1

    @staticmethod
    def w_fit(lambda_A, h_arcsec):
        """
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        """
        assert lambda_A >= 3800, "w_fit is only valid for lambda_A >= 3800"
        assert lambda_A <= 12000, "w_fit is not tested for lambda_A >= 12000"
        assert h_arcsec >= 0, "h_arcsec must be non-negative"
        assert h_arcsec <= 50, ""
        return 0.02 + h_arcsec**0.6 * 0.0175 + 4e2 / (lambda_A - 1600 + h_arcsec * 20)

    def fill_NLTE_n_w_parametrized(self, h_arcsec):
        """
        Reference: (12.1)
        Assume flat spectrum.
        (19) in A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        1'' = 725 km
        """
        assert self.df is None, "Cannot modify radiation tensor after it has been constructed"
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            lambda_ul_A = frequency_hz_to_lambda_A(nu_ul)

            J00 = self.n_fit(lambda_ul_A) * 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambda_ul_A, h_arcsec) / sqrt2

            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = delta(K, 0) * delta(Q, 0) * J00 + delta(K, 2) * delta(Q, 0) * J20
        self.construct_df()
        return self

    def get_NLTE_n_w_parametrized_stokes_I(self, h_arcsec, theta, nu):
        """
        Get Stokes I that is consistent with NLTE n and w JKQ tensor.
        I = J00 + 5 * J20 * P2(cos(theta))
        """
        stokesI = np.zeros_like(nu)
        for i, nui in enumerate(nu):  # Todo can be vectorized
            lambdai = frequency_hz_to_lambda_A(nui)
            J00 = self.n_fit(lambdai) * 2 * h_erg_s * nui**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambdai, h_arcsec) / sqrt2
            stokesI[i] = J00 + 5 * J20 * (3 * np.cos(theta) ** 2 - 1) / 2

        return stokesI

    def __call__(self, transition: Transition, K: int, Q: int) -> float:
        result = self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]
        return result

    def set(self, transition: Transition, K: int, Q: int, value):
        key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
        self.data[key] = value

    def construct_df(self):
        dfs = []
        for transition in self.transition_registry.transitions.values():
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                value = self.data[key]
                dfs.append(
                    pd.DataFrame(
                        {
                            "transition_id": transition.transition_id,
                            "K": K,
                            "Q": Q,
                            "radiation_tensor": value,
                        },
                        index=[0],
                    )
                )
        self.df = pd.concat(dfs, ignore_index=True)

    def rotate(self, D: WignerD):
        """
        (2.78), or more precisely, equation above (2.80)
        And also using the fact that J has K <= 2 for electric dipole transitions
        """

        new_J = RadiationTensor(transition_registry=self.transition_registry)
        for transition in self.transition_registry.transitions.values():
            for K, Q in nested_loops(
                K=FROMTO(0, 2),
                Q=PROJECTION("K"),
            ):
                new_J.set(
                    transition=transition,
                    K=K,
                    Q=Q,
                    value=summate(lambda P: self(transition=transition, K=K, Q=P) * D(K=K, P=P, Q=Q), P=PROJECTION(K)),
                )
        new_J.construct_df()
        return new_J

    def rotate_to_magnetic_frame(self, chi_B, theta_B):
        D = WignerD(alpha=chi_B, beta=theta_B, gamma=0, K_max=2)
        return self.rotate(D=D)
