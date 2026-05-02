try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python <3.11

from typing import Dict, Union

import pandas as pd

from solrat.atom_model.base_atom_model.object.radiation_tensor import BaseRadiationTensor
from solrat.atom_model.multi_level_atom_model.object.multi_level_atom_config import MultiLevelAtomConfig
from solrat.atom_model.multi_level_atom_model.object.transition_registry import Transition, TransitionRegistry
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.rotations import WignerD
from solrat.atom_model.shared.utility.constants import c_cm_sm1, h_erg_s, sqrt2
from solrat.atom_model.shared.utility.functions import frequency_sm1_to_lambda_A, get_planck_BP
from solrat.engine.functions.decorators import log_method
from solrat.engine.functions.general import delta, half_int_to_str
from solrat.engine.functions.looping import FROMTO, PROJECTION
from solrat.engine.generators.nested_loops import nested_loops
from solrat.engine.generators.summate import summate


class RadiationTensor(BaseRadiationTensor):
    r"""
    Radiation tensor :math:`J^K_Q(\nu_{ul})` for the Multi-Level atom model.

    Stored per transition. :math:`K \le 2` by construction for E1 transitions.

    :param transition_registry:  :class:`TransitionRegistry` instance.

    Reference: (LL04 5.157)
    """

    def __init__(self, transition_registry: TransitionRegistry):
        super().__init__()
        self.transition_registry = transition_registry
        self._df: Union[pd.DataFrame, None] = None
        self.data: Dict[str, float] = {}

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.construct_df()
        if self._df is None:
            raise RuntimeError("df has not been initialized")
        return self._df

    @classmethod
    def from_model_config(cls, config: MultiLevelAtomConfig) -> Self:
        return cls(transition_registry=config.transition_registry)

    @staticmethod
    def get_key(transition_id: str, K: int, Q: int) -> str:
        return f"{transition_id}_{half_int_to_str(K)}_{half_int_to_str(Q)}"

    @log_method
    def fill_planck(self, temperature_K: float) -> "RadiationTensor":
        r"""
        Flat-spectrum Planck approximation.
        """
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            planck = get_planck_BP(nu_sm1=nu_ul, temperature_K=temperature_K)
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = planck * delta(K, 0) * delta(Q, 0)
        self._df = None
        return self

    @staticmethod
    def n_fit(lambda_A: float) -> float:
        r"""
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542.
        """
        assert lambda_A >= 2750, "n_fit is only valid for lambda_A >= 2750"
        assert lambda_A <= 12000, "n_fit is not tested for lambda_A >= 12000"
        return 3e-10 * (lambda_A - 2500) ** 2.1

    @staticmethod
    def w_fit(lambda_A, h_arcsec) -> float:
        r"""
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542.
        """
        assert lambda_A >= 3800, "w_fit is only valid for lambda_A >= 3800"
        assert lambda_A <= 12000, "w_fit is not tested for lambda_A >= 12000"
        assert h_arcsec >= 0, "h_arcsec must be non-negative"
        assert h_arcsec <= 50, "w_fit is not tested for h_arcsec>50"
        return 0.02 + h_arcsec**0.6 * 0.0175 + 4e2 / (lambda_A - 1600 + h_arcsec * 20)

    @log_method
    def fill_NLTE_n_w_parametrized(self, h_arcsec) -> "RadiationTensor":
        r"""
        Fill the radiation tensor with the anisotropic {n, w} parametrization
        from A. Asensio Ramos et al (2008).
        """
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            lambda_ul_A = frequency_sm1_to_lambda_A(nu_ul)

            J00 = self.n_fit(lambda_ul_A) * 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambda_ul_A, h_arcsec) / sqrt2

            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = delta(K, 0) * delta(Q, 0) * J00 + delta(K, 2) * delta(Q, 0) * J20
        self._df = None
        return self

    def get(self, transition: Transition, K: int, Q: int) -> float:
        return self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]

    def get_from_transition_id(self, transition_id: str, K: int, Q: int) -> float:
        return self.data[self.get_key(transition_id=transition_id, K=K, Q=Q)]

    def set(self, transition: Transition, K: int, Q: int, value):
        key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
        self.data[key] = value
        self._df = None

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
        self._df = pd.concat(dfs, ignore_index=True)

    @log_method
    def rotate(self, D: WignerD) -> "RadiationTensor":
        r"""
        Rotate the :math:`J^K_Q` tensor.

        Reference: (LL04 2.78)
        """
        new_J = RadiationTensor(transition_registry=self.transition_registry)
        for transition in self.transition_registry.transitions.values():
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                new_J.set(
                    transition=transition,
                    K=K,
                    Q=Q,
                    value=summate(
                        lambda P: self.get(transition=transition, K=K, Q=P) * D(K=K, P=P, Q=Q), P=PROJECTION(K)
                    ),
                )
        return new_J

    @log_method
    def rotate_to_magnetic_frame(self, angles: Angles) -> "RadiationTensor":
        r"""
        Rotate :math:`J^K_Q` to the magnetic reference frame.
        """
        D = WignerD(alpha=angles.chi_B, beta=angles.theta_B, gamma=0, K_max=2)
        return self.rotate(D=D)
