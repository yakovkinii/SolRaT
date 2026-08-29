import logging
import unittest

import numpy as np

from solrat.atom_model.model_registry import Models
from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry
from solrat.atom_model.multi_level_atom_model.object.transition_registry import TransitionRegistry
from solrat.atom_model.shared.common_api.stratified_nlte_atmosphere import (
    PrescribedRadiationStratifiedAtmosphere,
    StratifiedAtmosphere,
)
from solrat.atom_model.shared.object.stokes import Stokes
from solrat.atom_model.shared.utility.functions import (
    energy_cmm1_to_frequency_sm1,
    frequency_sm1_to_lambda_A,
    get_frequencies_from_air_wavelength_range,
    lambda_vacuum_to_air,
)
from solrat.atom_model.shared.utility.log_setup import setup_logging
from solrat.engine.functions.special import pseudo_hash

N_DEPTH = 3


def build_two_level_model():
    r"""
    A J=0 -> J=1 resonance line as a multi-level atom.
    """
    level_registry = LevelRegistry()
    level_registry.register_level(alpha="1s", J=0, energy_cmm1=0, g=1.0)
    level_registry.register_level(alpha="2p", J=1, energy_cmm1=20_000, g=1.2)

    transition_registry = TransitionRegistry()
    transition_registry.register_transition(
        level_upper=level_registry.get_level(alpha="2p", J=1),
        level_lower=level_registry.get_level(alpha="1s", J=0),
        einstein_a_ul_sm1=1e7,
    )

    nu_ul = energy_cmm1_to_frequency_sm1(20_000)
    reference_lambda_A_air = lambda_vacuum_to_air(frequency_sm1_to_lambda_A(nu_ul))

    model = Models.multi_level_atom()
    model = model.configure(
        config=model.Config(
            level_registry=level_registry,
            transition_registry=transition_registry,
            atomic_mass_amu=4.0,
            reference_lambda_A_air=reference_lambda_A_air,
        )
    )
    return model, reference_lambda_A_air


def tiny_nu(reference_lambda_A_air):
    return get_frequencies_from_air_wavelength_range(
        lower_wavelength_A=reference_lambda_A_air - 0.3,
        upper_wavelength_A=reference_lambda_A_air + 0.3,
        step_A=5e-2,
    )


def thin_stratification(model):
    return StratifiedAtmosphere(
        model=model,
        height_cm=np.linspace(0.0, 150e5, N_DEPTH),
        temperature_K=6000.0,
        number_density_cm3=1.0e11,
        magnetic_field_gauss=100.0,
        theta_B=0.3,
        delta_v_turbulent_cm_sm1=2.0e5,
        continuum_opacity_cm_m1=1e-10,
    )


def anisotropic_tensor(model):
    return model.RadiationTensor.from_model_config(model.config).fill_NLTE_n_w_allen(h_arcsec=30)


class TestPrescribedRadiationStratifiedAtmosphere(unittest.TestCase):
    r"""
    Height-stratified synthesis with a prescribed :math:`J^K_Q` (no Lambda-iteration). Each variant
    exercises a different way of supplying the tensor and the imposed source function, reduced to a
    ``pseudo_hash`` of the emergent Stokes vector and locked against a baseline.
    """

    def _run_and_lock(self, name, atmosphere, nu, previous_hash):
        emergent = atmosphere.forward(initial_stokes=Stokes.from_BP(nu_sm1=nu, temperature_K=5700))
        new_hash = pseudo_hash(emergent.I, emergent.Q, emergent.U, emergent.V)
        logging.info(f"{name} current={new_hash!r} previous={previous_hash!r}")
        assert np.isfinite(new_hash)
        assert np.isclose(new_hash, previous_hash, rtol=1e-8, atol=0.0)

    def test_callable_tensor_scalar_source_delo_linear(self):
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)
        atmosphere = PrescribedRadiationStratifiedAtmosphere(
            model=model,
            stratification=thin_stratification(model),
            radiation_tensor=lambda i, z, tau: anisotropic_tensor(model),
            los_theta=0.4,
            source_function_I=1.0,
            transfer_scheme="delo_linear",
        )
        self._run_and_lock("test_callable_tensor_scalar_source_delo_linear", atmosphere, nu,
                           previous_hash=3.3519554389221464)

    def test_sequence_tensor_array_source(self):
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)
        atmosphere = PrescribedRadiationStratifiedAtmosphere(
            model=model,
            stratification=thin_stratification(model),
            radiation_tensor=[anisotropic_tensor(model) for _ in range(N_DEPTH)],
            los_theta=0.3,
            source_function_I=np.linspace(1.0, 0.5, N_DEPTH),
        )
        self._run_and_lock("test_sequence_tensor_array_source", atmosphere, nu, previous_hash=1.9013646368429222)

    def test_single_tensor_callable_source_tangential(self):
        setup_logging()
        model, reference_lambda_A_air = build_two_level_model()
        nu = tiny_nu(reference_lambda_A_air)
        atmosphere = PrescribedRadiationStratifiedAtmosphere(
            model=model,
            stratification=thin_stratification(model),
            radiation_tensor=anisotropic_tensor(model),
            los_theta=np.pi / 2,  # tangential ray -> emergent from the surface node
            source_function_I=lambda z, tau_c: 1.0 + 0.5 * tau_c,
        )
        self._run_and_lock("test_single_tensor_callable_source_tangential", atmosphere, nu, previous_hash=3.4)


if __name__ == "__main__":
    unittest.main()
