import logging
from typing import Optional, Union

import numpy as np
from numpy import real

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import BaseAtmosphereParameters
from solrat.atom_model.base_atom_model.object.radiation_tensor import BaseRadiationTensor
from solrat.atom_model.base_atom_model.radiative_transfer_equations import BaseRTE
from solrat.atom_model.model_registry import Model
from solrat.atom_model.shared.object.angles import Angles
from solrat.atom_model.shared.object.radiative_transfer_coefficients import RadiativeTransferCoefficients
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import get_planck_BP
from solrat.engine.functions.decorators import log_method


class MilneEddingtonSlabAtmosphere:
    r"""
    Semi-infinite atmosphere with a constant propagation matrix and an LTE source function linear
    in the continuum optical depth -- the Milne-Eddington model.

    This is the sibling of :class:`ConstantPropertySlabAtmosphere`: the constant-property slab is a
    *finite* slab with a *constant* source solved by the DELO/evolution-operator step, whereas this
    atmosphere is *unbounded from below* with a source that varies *linearly* with depth,

    .. math::

        S(\tau_c) = S_0 + S_1\, \tau_c .

    For a constant propagation matrix and a linear source the polarized transfer equation has the
    closed-form Unno-Rachkovsky solution

    .. math::

        I(0) = S_0\, \mathbf u + S_1\, \hat K^{-1} \mathbf u, \qquad \mathbf u = (1, 0, 0, 0)^T,

    with the total propagation matrix normalized to the continuum,
    :math:`\hat K = \mathbf 1 + \eta_0\, K_\tau^{\mathrm{line}}`, where :math:`\eta_0` is the
    line-center-to-continuum opacity ratio and :math:`K_\tau^{\mathrm{line}}` is the line
    propagation matrix normalized so that its line-center :math:`\eta_I = 1` (i.e. ``rtc.K_tau()``).
    Optical depth is measured along the line of sight, matching the convention of
    :class:`ConstantPropertySlabAtmosphere` (no explicit :math:`\mu`).

    Because the "+1" continuum on the diagonal keeps :math:`\hat K` non-singular everywhere, no
    pseudo-inverse fallback is needed. This is the standard forward model behind Milne-Eddington
    Stokes inversions and the exact analytic reference for LTE Zeeman line transfer.

    The Milne-Eddington source is the LTE Planck function, taken equal for line and continuum and
    linear in the continuum optical depth (:math:`S_{\rm line} = S_{\rm cont} = B_{\rm P}`, LL04
    eq. 9.105): a thermal, unpolarized, depth-linear source into which the radiation field does not
    enter. For an LTE SEE the density matrix -- and hence :math:`\hat K` -- is likewise independent
    of the radiation tensor, so ``radiation_tensor`` has no effect on the result and is accepted only
    for interface compatibility with :class:`ConstantPropertySlabAtmosphere`. A non-LTE SEE would
    instead make :math:`\hat K` depend on the radiation tensor through the atomic polarization.

    :param model: Model instance.
    :param radiation_tensor: RadiationTensor instance (unused for an LTE SEE; see above).
    :param atmosphere_parameters: AtmosphereParameters instance.
    :param angles: Angles instance.
    :param line_to_continuum_ratio: line-center-to-continuum opacity ratio :math:`\eta_0` (> 0).
    :param source_gradient: source-function gradient :math:`S_1 = \mathrm dS/\mathrm d\tau_c` along
        the line of sight, in specific-intensity units [erg s^-1 cm^-2 Hz^-1 sr^-1] (a positive value
        gives an absorption line).
    :param source_surface: surface source :math:`S_0` in the same specific-intensity units; if
        ``None``, defaults to the Planck function at the atmosphere temperature, averaged over the
        frequency grid (Milne-Eddington treats the source as constant across the narrow line window).

    Reference: Unno (1956); Rachkovsky (1962); LL04 Sec. 9.8 (eqs. 9.105, 9.109);
    del Toro Iniesta (2003), Ch. 9.
    """

    def __init__(
        self,
        model: Model,
        radiation_tensor: BaseRadiationTensor,
        atmosphere_parameters: BaseAtmosphereParameters,
        angles: Angles,
        line_to_continuum_ratio: float,
        source_gradient: float,
        source_surface: Optional[float] = None,
    ):
        assert line_to_continuum_ratio > 0, "line_to_continuum_ratio must be positive."
        self.model = model
        self.radiation_tensor = radiation_tensor
        self.atmosphere_parameters = atmosphere_parameters
        self.angles = angles
        self.line_to_continuum_ratio = line_to_continuum_ratio
        self.source_gradient = source_gradient
        self.source_surface = source_surface
        self.see = model.StatisticalEquilibriumEquations.from_model_config(config=model.config)
        self._rte: Union[BaseRTE, None] = None
        self._rtc: Union[RadiativeTransferCoefficients, None] = None

    @property
    def rte(self) -> BaseRTE:
        if self._rte is None:
            raise RuntimeError("rte has not been initialized")  # pragma: no cover
        return self._rte

    @property
    def rtc(self) -> RadiativeTransferCoefficients:
        if self._rtc is None:
            raise RuntimeError("rtc has not been initialized")  # pragma: no cover
        return self._rtc

    @log_method
    def forward(self, initial_stokes: Stokes) -> Stokes:
        r"""
        Emergent Stokes vector of the semi-infinite Milne-Eddington atmosphere.

        Only the frequency grid ``initial_stokes.nu`` is used: a semi-infinite atmosphere has no
        illuminated lower boundary, so the incident Stokes vector itself is irrelevant. The
        argument is kept for interface compatibility with :class:`ConstantPropertySlabAtmosphere`.
        """
        logging.info("Processing a Milne-Eddington (semi-infinite) slab...")
        nu = initial_stokes.nu

        self.see.fill_all_equations(
            atmosphere_parameters=self.atmosphere_parameters,
            radiation_tensor_in_magnetic_frame=self.radiation_tensor.rotate_to_magnetic_frame(angles=self.angles),
        )
        rho = self.see.get_solution()

        self._rte = self.model.RadiativeTransferEquations.from_model_config(config=self.model.config, nu=nu)
        self._rtc = self.rte.calculate_all_coefficients(
            atmosphere_parameters=self.atmosphere_parameters,
            angles=self.angles,
            rho=rho,
        )

        # Total propagation matrix in continuum optical-depth units: Khat = 1 + eta_0 K_tau_line.
        # rtc.K_tau() is the line matrix normalized to line-center eta_I = 1; the "+1" is the
        # (unpolarized) continuum, so Khat is diagonally dominant and always invertible.
        eta_0 = self.line_to_continuum_ratio
        K_hat = eta_0 * self.rtc.K_tau()  # [Nnu, 4, 4]
        for i in range(4):
            K_hat[:, i, i] += 1.0

        # LTE source S(tau_c) = S0 + S1 tau_c (linear Planck, LL04 eq. 9.105), taken constant across
        # the narrow line window.
        if self.source_surface is None:
            source_0 = float(np.mean(get_planck_BP(nu_sm1=nu, temperature_K=self.atmosphere_parameters.temperature_K)))
        else:
            source_0 = float(self.source_surface)
        source_1 = float(self.source_gradient)

        # Unno-Rachkovsky closed form (LL04 eq. 9.109): I(0) = S0 u + S1 Khat^{-1} u,
        # u = (1, 0, 0, 0)^T. The
        # right-hand side carries an explicit trailing axis so numpy>=2.0 reads it as a stack of
        # column vectors rather than a stack of matrices.
        u = np.zeros((len(nu), 4, 1), dtype=np.float64)
        u[:, 0, 0] = 1.0
        k_inv_u = np.linalg.solve(K_hat, u)[:, :, 0]  # [Nnu, 4]
        e0 = np.array([1.0, 0.0, 0.0, 0.0])
        emergent = source_0 * e0[np.newaxis, :] + source_1 * k_inv_u  # [Nnu, 4]

        logging.info("Completed processing a Milne-Eddington slab")

        return Stokes(
            nu=nu,
            I=real(emergent[:, 0]),
            Q=real(emergent[:, 1]),
            U=real(emergent[:, 2]),
            V=real(emergent[:, 3]),
        )
