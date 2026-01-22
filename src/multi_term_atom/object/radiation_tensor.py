from typing import Dict, Union

import numpy as np
import pandas as pd

from src.common.constants import c_cm_sm1, h_erg_s, sqrt2
from src.common.functions import frequency_hz_to_lambda_A, get_planck_BP
from src.common.rotations import WignerD
from src.engine.functions.general import delta, half_int_to_str
from src.engine.functions.looping import FROMTO, PROJECTION
from src.engine.generators.nested_loops import nested_loops
from src.engine.generators.summate import summate
from src.multi_term_atom.terms_levels_transitions.transition_registry import (
    Transition,
    TransitionRegistry,
)


class RadiationTensor:
    def __init__(self, transition_registry: TransitionRegistry):
        """
        Radiation tensor J^K_Q (nu_ul).
        Here we assume that transitions are spread apart in frequency, so that we can assign a bijection of
        transition <-> nu_ul, and store J for each transition instead for clarity.
        K is always <=2 by construction (see eg. 5.157) for electric-dipole transitions due to T tensor.

        :param transition_registry: TransitionRegistry instance

        Reference: (5.157)
        """
        super().__init__()
        self.transition_registry = transition_registry
        self.df: Union[pd.DataFrame, None] = None
        self.data: Dict[str, Union[float, np.ndarray]] = {}

    def get_df(self) -> pd.DataFrame:
        """
        Get the dataframe representation of the JKQ radiation tensor.
        """
        if self.df is None:
            self.construct_df()
        return self.df

    @staticmethod
    def get_key(transition_id: str, K: int, Q: int) -> str:
        return f"{transition_id}_{half_int_to_str(K)}_{half_int_to_str(Q)}"

    def fill_planck(self, T_K: float) -> "RadiationTensor":
        """
        Flat-spectrum approximation, i.e. J needs to be defined for each transition, not for each frequency.

        :param T_K: Temperature in Kelvin
        :return:
        """
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            planck = get_planck_BP(nu_sm1=nu_ul, T_K=T_K)
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = planck * delta(K, 0) * delta(Q, 0)
        self.df = None
        return self

    @staticmethod
    def n_fit(lambda_A: float) -> float:
        """
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433

        :param lambda_A: wavelength in Angstrom
        """
        assert lambda_A >= 2750, "n_fit is only valid for lambda_A >= 2750"
        assert lambda_A <= 12000, "n_fit is not tested for lambda_A >= 12000"
        return 3e-10 * (lambda_A - 2500) ** 2.1

    @staticmethod
    def w_fit(lambda_A, h_arcsec) -> float:
        """
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433

        :param lambda_A: wavelength in Angstrom
        :param h_arcsec: height above the Sun's surface in arcsec
        """
        assert lambda_A >= 3800, "w_fit is only valid for lambda_A >= 3800"
        assert lambda_A <= 12000, "w_fit is not tested for lambda_A >= 12000"
        assert h_arcsec >= 0, "h_arcsec must be non-negative"
        assert h_arcsec <= 50, "w_fit is not tested for h_arcsec>50"
        return 0.02 + h_arcsec**0.6 * 0.0175 + 4e2 / (lambda_A - 1600 + h_arcsec * 20)

    def fill_NLTE_n_w_parametrized(self, h_arcsec) -> "RadiationTensor":
        """
        Fill the radiation tensor with an anisotropic parametrization from A. Asensio Ramos et al (2008)
        Assume flat spectrum. Note that this fit is a smooth fit of data in A. Asensio Ramos et al (2008).

        :param h_arcsec: height above the Sun's surface in arcsec; 1'' = 725 km

        Reference: (12.1)
        Reference: Figures and eq. (19) in
        A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        """
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            lambda_ul_A = frequency_hz_to_lambda_A(nu_ul)

            J00 = self.n_fit(lambda_ul_A) * 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambda_ul_A, h_arcsec) / sqrt2

            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = delta(K, 0) * delta(Q, 0) * J00 + delta(K, 2) * delta(Q, 0) * J20
        self.df = None
        return self

    def get_NLTE_n_w_parametrized_stokes_I(self, h_arcsec, theta, nu):
        """
        Get Stokes I that is consistent with the anisotropic {n, w} JKQ tensor.
        I = J00 + 5 * J20 * P2(cos(theta))

        :param h_arcsec: height above the Sun's surface in arcsec; 1'' = 725 km
        :param theta: theta angle (see the RTE geometry explanation).
        :param nu: frequency in [1/cm]

        Reference: (5.164)
        """
        stokesI = np.zeros_like(nu)
        for i, nui in enumerate(nu):
            lambdai = frequency_hz_to_lambda_A(nui)
            J00 = self.n_fit(lambdai) * 2 * h_erg_s * nui**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambdai, h_arcsec) / sqrt2
            stokesI[i] = J00 + 5 * J20 * (3 * np.cos(theta) ** 2 - 1) / 2

        return stokesI

    def get(self, transition: Transition, K: int, Q: int) -> float:
        """
        Get the component of the JKQ radiation tensor for the specified transition.
        """
        return self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]

    def set(self, transition: Transition, K: int, Q: int, value):
        """
        Set the component of the JKQ radiation tensor for the specified transition.
        """
        key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
        self.data[key] = value
        self.df = None

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

    def rotate(self, D: WignerD) -> "RadiationTensor":
        """
        Rotate the JKQ tensor according to the D rotation.

        Reference: (2.78), or more precisely, equation above (2.80)
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
                    value=summate(
                        lambda P: self.get(transition=transition, K=K, Q=P) * D(K=K, P=P, Q=Q), P=PROJECTION(K)
                    ),
                )
        return new_J

    def rotate_to_magnetic_frame(self, chi_B, theta_B) -> "RadiationTensor":
        """
        Rotate JKQ to the magnetic reference frame.

        :param chi_B: Magnetic field angle chi.
        :param theta_B: Magnetic field angle theta.
        """
        D = WignerD(alpha=chi_B, beta=theta_B, gamma=0, K_max=2)
        return self.rotate(D=D)
