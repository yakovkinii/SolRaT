import numpy as np

from solrat.atom_model.base_atom_model.object.atmosphere_parameters import (
    BaseAtmosphereParameters,
)
from solrat.atom_model.multi_term_atom_model.object.multi_term_atom_config import (
    MultiTermAtomConfig,
)
from solrat.atom_model.shared.utility.constants import atomic_mass_unit_g, kB_erg_Km1
from solrat.atom_model.shared.utility.functions import nu_larmor


class AtmosphereParameters(BaseAtmosphereParameters):
    r"""
    A container for all atmosphere parameters needed in RTE/SEE.

    :param magnetic_field_gauss: :math:`B` [G]
    :param temperature_K: :math:`T` [K]
    :param delta_v_turbulent_cm_sm1: turbulent microscopic velocity [cm/s]
    :param macroscopic_velocity_cm_sm1: macroscopic velocity [cm/s]
    :param voigt_a: Voigt :math:`a` parameter.
    """

    def __init__(
        self,
        model_config: MultiTermAtomConfig,
        magnetic_field_gauss,
        temperature_K,
        delta_v_turbulent_cm_sm1=0,
        macroscopic_velocity_cm_sm1=0,
        voigt_a=0,
    ):
        self.magnetic_field_gauss = magnetic_field_gauss
        self.temperature_K = temperature_K
        self.delta_v_thermal_cm_sm1 = np.sqrt(
            delta_v_turbulent_cm_sm1**2
            + 2 * kB_erg_Km1 * temperature_K / model_config.atomic_mass_amu / atomic_mass_unit_g
        )
        self.nu_larmor = nu_larmor(magnetic_field_gauss=magnetic_field_gauss)
        self.macroscopic_velocity_cm_sm1 = macroscopic_velocity_cm_sm1
        self.voigt_a = voigt_a
