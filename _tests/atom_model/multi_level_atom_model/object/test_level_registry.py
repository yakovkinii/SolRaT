import unittest

from solrat.atom_model.multi_level_atom_model.object.level_registry import LevelRegistry


class TestLevelRegistryLSCoupling(unittest.TestCase):
    r"""
    Registering a level with the Lande factor derived from LS coupling.
    """

    def test_J0_has_zero_g(self):
        registry = LevelRegistry()
        registry.register_level_LS_coupling(alpha="a", L=0, S=0, J=0, energy_cmm1=0.0)
        assert registry.get_level(alpha="a", J=0).g == 0

    def test_triplet_P2_lande_factor(self):
        # ^3P_2 (L=1, S=1, J=2): g = 1 + [J(J+1)+S(S+1)-L(L+1)] / (2 J (J+1)) = 3/2.
        registry = LevelRegistry()
        registry.register_level_LS_coupling(alpha="3P", L=1, S=1, J=2, energy_cmm1=100.0)
        assert abs(registry.get_level(alpha="3P", J=2).g - 1.5) < 1e-12


class TestLevelHashable(unittest.TestCase):
    r"""
    A :class:`Level` is hashable so it can key dictionaries/sets.
    """

    def test_level_is_hashable(self):
        registry = LevelRegistry()
        registry.register_level(alpha="a", J=1, energy_cmm1=1.0, g=1.0)
        level = registry.get_level(alpha="a", J=1)
        assert hash(level) == hash(level)
        assert len({level, level}) == 1


if __name__ == "__main__":
    unittest.main()
