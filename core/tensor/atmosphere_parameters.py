from core.utility.misc import nu_larmor


class AtmosphereParameters:
    def __init__(self, magnetic_field_gauss):
        self.nu_larmor = nu_larmor(magnetic_field_gauss=magnetic_field_gauss)
