from typing import Dict, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from solrat.atom_model.base_atom_model.object.radiation_tensor import (
    BaseRadiationTensor,
)
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import (
    MultiTermAtomConfig,
)
from solrat.atom_model.multi_term_atom_model.object.transition_registry import (
    Transition,
    TransitionRegistry,
)
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.rotations import WignerD
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, sqrt2
from solrat.atom_model.shared.utility.functions import (
    frequency_hz_to_lambda_A,
    get_planck_BP,
)
from solrat.engine.functions.general import delta, half_int_to_str
from solrat.engine.functions.looping import FROMTO, PROJECTION
from solrat.engine.generators.nested_loops import nested_loops
from solrat.engine.generators.summate import summate


class RadiationTensor(BaseRadiationTensor):
    def __init__(self, transition_registry: TransitionRegistry):
        r"""
        Radiation tensor :math:`J^K_Q(nu_ul)`.
        Here we assume that transitions are spread apart in frequency, so that we can assign a bijection of
        transition <-> :math:`\nu_{ul}`, and store :math:`J` for each transition instead for clarity.
        :math:`K` is always :math:`\le 2` by construction (see eg. 5.157) for electric-dipole transitions
        due to :math:`T` tensor.

        :param transition_registry: :any:`TransitionRegistry` instance

        Reference: (LL04 5.157)
        """
        super().__init__()
        self.transition_registry = transition_registry
        self.df: Union[pd.DataFrame, None] = None
        self.data: Dict[str, Union[float, np.ndarray]] = {}

    @classmethod
    def from_model_config(cls, config: MultiTermAtomConfig) -> Self:
        return cls(transition_registry=config.transition_registry)

    def get_df(self) -> pd.DataFrame:
        r"""
        Get the dataframe representation of the :math:`J^K_Q` radiation tensor.
        """
        if self.df is None:
            self.construct_df()
        return self.df

    @staticmethod
    def get_key(transition_id: str, K: int, Q: int) -> str:
        return f"{transition_id}_{half_int_to_str(K)}_{half_int_to_str(Q)}"

    def fill_planck(self, T_K: float) -> "RadiationTensor":
        r"""
        Flat-spectrum approximation, i.e. :math:`J^K_Q` needs to be defined for each transition,
        not for each frequency.

        :param T_K: Temperature in Kelvin
        :return: :any:`RadiationTensor` instance
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
        r"""
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433

        :param lambda_A: wavelength in Angstrom
        """
        assert lambda_A >= 2750, "n_fit is only valid for lambda_A >= 2750"
        assert lambda_A <= 12000, "n_fit is not tested for lambda_A >= 12000"
        return 3e-10 * (lambda_A - 2500) ** 2.1

    @staticmethod
    def w_fit(lambda_A, h_arcsec) -> float:
        r"""
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
        r"""
        Fill the radiation tensor with an anisotropic parametrization from A. Asensio Ramos et al (2008)
        Assume flat spectrum. Note that this fit is a smooth fit of data in A. Asensio Ramos et al (2008).

        :param h_arcsec: height above the Sun's surface in arcsec; 1'' = 725 km

        Reference: (LL04 12.1)
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
        r"""
        Get Stokes I that is consistent with the anisotropic {n, w} :math:`J^K_Q` tensor.

        .. math::
            I = J^0_0 + 5 J^2_0 P_2(\cos(\theta))

        :param h_arcsec: height above the Sun's surface in arcsec; 1'' = 725 km
        :param theta: theta angle (see the RTE geometry explanation).
        :param nu: frequency in [1/s]

        Reference: (LL04 5.164)
        """
        stokesI = np.zeros_like(nu)
        for i, nui in enumerate(nu):
            lambdai = frequency_hz_to_lambda_A(nui)
            J00 = self.n_fit(lambdai) * 2 * h_erg_s * nui**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambdai, h_arcsec) / sqrt2
            stokesI[i] = J00 + 5 * J20 * (3 * np.cos(theta) ** 2 - 1) / 2

        return stokesI

    def get(self, transition: Transition, K: int, Q: int) -> float:
        r"""
        Get the component of the :math:`J^K_Q` radiation tensor for the specified transition.
        """
        return self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]

    def set(self, transition: Transition, K: int, Q: int, value):
        r"""
        Set the component of the :math:`J^K_Q` radiation tensor for the specified transition.
        """
        key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
        self.data[key] = value
        self.df = None

    def construct_df(self):
        r"""
        Construct the dataframe representation of the :math:`J^K_Q` tensor
        """
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
        r"""
        Rotate the :math:`J^K_Q` tensor according to the :math:`D` rotation.

        Reference: (LL04 2.78), or more precisely, equation above (LL04 2.80)
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

    def rotate_to_magnetic_frame(self, angles: Angles) -> Self:
        r"""
        Rotate :math:`J^K_Q` to the magnetic reference frame.

        :param angles: :any:`Angles` instance with observation geometry.
        """
        D = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
        return self.rotate(D=D)
