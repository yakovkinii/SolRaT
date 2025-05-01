from core.utility.misc import nu_larmor


class AtmosphereParameters:
    def __init__(self, magnetic_field_gauss, delta_v_thermal_cm_sm1):
        self.magnetic_field_gauss = magnetic_field_gauss
        self.delta_v_thermal_cm_sm1 = delta_v_thermal_cm_sm1
        self.nu_larmor = nu_larmor(magnetic_field_gauss=magnetic_field_gauss)
