import pandas as pd

from src.core.engine.functions.decorators import log_function_not_tested
from src.core.engine.functions.general import delta
from src.core.engine.functions.looping import FROMTO, PROJECTION
from src.core.engine.generators.nested_loops import nested_loops
from src.core.engine.objects.container import Container
from src.core.physics.constants import c_cm_sm1, h_erg_s, sqrt2
from src.core.physics.functions import frequency_hz_to_lambda_A, get_planck_BP
from src.two_term_atom.terms_levels_transitions.transition_registry import Transition, TransitionRegistry


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
        self.df: pd.DataFrame = None

    def fill_planck(self, T_K: float):
        """
        Flat-spectrum approximation, i.e. J needs to be defined for each transition, not for each frequency.
        """
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            planck = get_planck_BP(nu_sm1=nu_ul, T_K=T_K)
            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = planck * delta(K, 0) * delta(Q, 0)
        self.construct_df()
        return self

    @staticmethod
    @log_function_not_tested
    def n_fit(lambda_A):
        """
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        """
        return 1e-25 * lambda_A**5.83

    @staticmethod
    @log_function_not_tested
    def w_fit(lambda_A, h_arcsec):
        """
        Fit from Fig 4 of A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        """
        w0 = 0.19 + 0.0035 * h_arcsec
        alpha = 0.90 - 0.013 * h_arcsec
        return w0 * (lambda_A / 4000.0) ** (-alpha)

    def fill_NLTE_w(self, h_arcsec):
        """
        Reference: (12.1)
        Assume flat spectrum.
        (19) in A. Asensio Ramos et al 2008 ApJ 683 542 https://iopscience.iop.org/article/10.1086/589433
        1'' = 725 km
        """
        for transition in self.transition_registry.transitions.values():
            nu_ul = transition.get_mean_transition_frequency_sm1()
            lambda_ul_A = frequency_hz_to_lambda_A(nu_ul)

            J00 = self.n_fit(h_arcsec) * 2 * h_erg_s * nu_ul**3 / c_cm_sm1**2
            J20 = J00 * self.w_fit(lambda_ul_A, h_arcsec) / sqrt2

            for K, Q in nested_loops(K=FROMTO(0, 2), Q=PROJECTION("K")):
                key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
                self.data[key] = delta(K, 0) * delta(Q, 0) * J00 + delta(K, 2) * delta(Q, 0) * J20
        self.construct_df()
        return self

    def __call__(self, transition: Transition, K: int, Q: int) -> float:
        result = self.data[self.get_key(transition_id=transition.transition_id, K=K, Q=Q)]
        return result

    def get(self, transition_id: str, K: int, Q: int) -> float:
        result = self.data[self.get_key(transition_id=transition_id, K=K, Q=Q)]
        return result

    def add(self, transition: Transition, K: int, Q: int, value):
        key = self.get_key(transition_id=transition.transition_id, K=K, Q=Q)
        if key in self.data:
            self.data[key] += value
        else:
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
