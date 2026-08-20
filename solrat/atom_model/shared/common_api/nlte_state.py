import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from solrat.atom_model.base_atom_model.object.rho import BaseRho


class NLTEState:
    r"""
    Reusable self-consistent NLTE state: the per-depth statistical tensors :math:`\rho^K_Q` and the
    height grid they live on.

    The output of :meth:`NLTEStratifiedAtmosphere.forward`, it can seed another run (resampled onto a
    different height grid), be chained one :math:`\Lambda`-iteration at a time, and be saved to disk.
    """

    def __init__(
        self,
        height_cm: Sequence[float],
        coherence_keys: Sequence[str],
        values: np.ndarray,
        model_signature: str = "",
    ):
        z = np.asarray(height_cm, dtype=np.float64)
        vals = np.asarray(values, dtype=np.complex128)
        assert z.ndim == 1 and len(z) >= 1, "height_cm must be a 1-D grid."
        assert np.all(np.diff(z) > 0) if len(z) > 1 else True, "height_cm must be strictly increasing."
        assert vals.shape == (len(z), len(coherence_keys)), "values must have shape [n_depth, n_keys]."
        self.height_cm = z
        self.coherence_keys = list(coherence_keys)
        self.values = vals
        self.model_signature = str(model_signature)

    @property
    def n_depth(self) -> int:
        return len(self.height_cm)

    @classmethod
    def from_rho_grid(
        cls, height_cm: Sequence[float], rho_grid: List[BaseRho], model_signature: str = ""
    ) -> "NLTEState":
        r"""
        Capture a per-depth density-matrix grid as a state.
        """
        z = np.asarray(height_cm, dtype=np.float64)
        assert len(rho_grid) == len(z), "rho_grid length must match the height grid."
        keys = sorted(rho_grid[0].data)
        values = np.array([[complex(rho.data.get(key, 0.0)) for key in keys] for rho in rho_grid], dtype=np.complex128)
        return cls(height_cm=z, coherence_keys=keys, values=values, model_signature=model_signature)

    def interpolate_to(self, height_cm: Sequence[float]) -> "NLTEState":
        r"""
        Resample onto a new height grid by linear interpolation of each coherence in log depth below
        the observer surface (:math:`\rho` is smooth in log optical depth and the grids are
        log-refined toward the surface). Values outside the original range are held at the endpoint.
        """
        z_new = np.asarray(height_cm, dtype=np.float64)
        if z_new.shape == self.height_cm.shape and np.allclose(z_new, self.height_cm):
            return self
        # Depth below the observer surface z[-1] (0 at the surface); rho is smooth in its logarithm.
        depth_old = self.height_cm[-1] - self.height_cm
        positive = depth_old[depth_old > 0]
        floor = 0.1 * float(positive.min()) if positive.size else 1.0  # keeps the surface node's log finite
        xi_old = np.log10(np.maximum(depth_old, floor))
        xi_new = np.log10(np.maximum(self.height_cm[-1] - z_new, floor))
        order = np.argsort(xi_old)  # np.interp needs an ascending sample coordinate
        real = np.empty((len(z_new), len(self.coherence_keys)), dtype=np.float64)
        imag = np.empty_like(real)
        for j in range(len(self.coherence_keys)):
            real[:, j] = np.interp(xi_new, xi_old[order], self.values[order, j].real)
            imag[:, j] = np.interp(xi_new, xi_old[order], self.values[order, j].imag)
        return NLTEState(
            height_cm=z_new,
            coherence_keys=self.coherence_keys,
            values=real + 1j * imag,
            model_signature=self.model_signature,
        )

    def apply_to_templates(self, templates: List[BaseRho]) -> None:
        r"""
        Overwrite the solver-visible values (``rho.data``) of structurally correct template density
        matrices in place, depth by depth, from this state.

        The templates set the coherence structure and this state supplies the numbers; resample with
        :meth:`interpolate_to` first if the grids differ.
        """
        assert len(templates) == self.n_depth, (
            f"Template grid has {len(templates)} depths but the state has {self.n_depth}; "
            "interpolate_to the template height grid first."
        )
        key_index = {key: j for j, key in enumerate(self.coherence_keys)}
        for i, template in enumerate(templates):
            for key in template.data:
                j = key_index.get(key)
                if j is not None:
                    template.data[key] = complex(self.values[i, j])
            # The vectorized get_vector() path (multi-term) reads a cached dataframe of the old
            # values; drop it so any later read rebuilds from data.
            if hasattr(template, "_datarows_df"):
                template._datarows_df = None

    def to_dicts(self) -> List[Dict[str, complex]]:
        r"""
        Per-depth ``{coherence_key: value}`` dictionaries.
        """
        return [
            {key: complex(self.values[i, j]) for j, key in enumerate(self.coherence_keys)} for i in range(self.n_depth)
        ]

    def save(self, path: str) -> None:
        r"""
        Write to a ``.npz`` file (no pickling).
        """
        np.savez(
            path,
            height_cm=self.height_cm,
            coherence_keys=np.array(self.coherence_keys),
            values_real=self.values.real,
            values_imag=self.values.imag,
            model_signature=np.array(self.model_signature),
        )

    @classmethod
    def load(cls, path: str) -> "NLTEState":
        r"""
        Read a state written by :meth:`save`.
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                height_cm=data["height_cm"],
                coherence_keys=[str(key) for key in data["coherence_keys"]],
                values=data["values_real"] + 1j * data["values_imag"],
                model_signature=str(data["model_signature"]),
            )

    def check_compatible(self, model_signature: str, coherence_keys: Optional[Sequence[str]] = None) -> None:
        r"""
        Warn on a model-signature mismatch and assert the coherence keys overlap, so a state from a
        different atom is not silently applied.
        """
        if model_signature and self.model_signature and model_signature != self.model_signature:
            logging.warning(
                "NLTE warm-start state was produced by model '%s' but is applied to '%s'.",
                self.model_signature,
                model_signature,
            )
        if coherence_keys is not None:
            assert set(self.coherence_keys) & set(coherence_keys), (
                "NLTE warm-start state shares no coherence keys with the target model; "
                "it belongs to a different atom."
            )
