from src.common.functions import nu_larmor


class AtmosphereParameters:
    def __init__(self, magnetic_field_gauss, delta_v_thermal_cm_sm1, macroscopic_velocity_cm_sm1=0, voigt_a=0):
        self.magnetic_field_gauss = magnetic_field_gauss
        self.delta_v_thermal_cm_sm1 = delta_v_thermal_cm_sm1
        self.nu_larmor = nu_larmor(magnetic_field_gauss=magnetic_field_gauss)
        self.macroscopic_velocity_cm_sm1 = macroscopic_velocity_cm_sm1
        self.voigt_a = voigt_a
