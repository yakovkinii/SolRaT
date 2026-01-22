import numpy as np

from src.common.constants import atomic_mass_unit_g, kB_erg_Km1
from src.common.functions import nu_larmor


class AtmosphereParameters:
    def __init__(
        self,
        magnetic_field_gauss,
        temperature_K,
        atomic_mass_au,
        delta_v_turbulent_cm_sm1=0,
        macroscopic_velocity_cm_sm1=0,
        voigt_a=0,
    ):
        """
        A container for all atmosphere parameters needed in RTE/SEE.

        :param magnetic_field_gauss: B [G]
        :param temperature_K: T [K]
        :param atomic_mass_au: M [atomic mass units]
        :param delta_v_turbulent_cm_sm1: turbulent microscopic velocity [cm/s]
        :param macroscopic_velocity_cm_sm1: macroscopic velocity [cm/s]
        :param voigt_a: Voigt a parameter.
        """
        self.magnetic_field_gauss = magnetic_field_gauss
        self.temperature_K = temperature_K
        self.delta_v_thermal_cm_sm1 = np.sqrt(
            delta_v_turbulent_cm_sm1**2 + 2 * kB_erg_Km1 * temperature_K / atomic_mass_au / atomic_mass_unit_g
        )
        self.nu_larmor = nu_larmor(magnetic_field_gauss=magnetic_field_gauss)
        self.macroscopic_velocity_cm_sm1 = macroscopic_velocity_cm_sm1
        self.voigt_a = voigt_a
