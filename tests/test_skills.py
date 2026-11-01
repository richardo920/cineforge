"""Offline source provenance, review and immutable shot binding acceptance."""

import unittest
from dataclasses import replace

from cineforge.contracts import ContractError, Shot
from cineforge.devices import Backend, GPUDevice
from cineforge.skills import (
    CatalogError,
    Compatibility,
    Permissions,
    ShotSkills,
    SkillPin,
    SourcePin,
    bind_shot,
)


def shot() -> Shot:
    return Shot("shot", "scene", 1, "synthetic", 42, 2.0625, "model", "a" * 40, "b" * 64)


class SourceTests(unittest.TestCase):
    def test_immutable_source_roundtrip(self) -> None:
        source = SourcePin("https://github.com/example/creative", "a" * 40, "SKILL.md", "b" * 64)
        self.assertEqual(SourcePin.from_json(source.to_json()), source)
        self.assertIn("/blob/" + "a" * 40, source.uri)
        for revision in ("main", "v1", "a" * 39):
            with self.subTest(revision=revision), self.assertRaises(CatalogError):
                replace(source, revision=revision)
        for path in ("../SKILL.md", "/tmp/skill", "a/../b", "a//b", "%2e%2e/a", "a\\b"):
            with self.subTest(path=path), self.assertRaises(CatalogError):
                replace(source, path=path)
        for repository in ("http://github.com/a/b", "https://user:pass@github.com/a/b",
                           "https://github.com/a/b?token=value", "https://github.com/a/b/"):
            with self.subTest(repository=repository), self.assertRaises(CatalogError):
                replace(source, repository=repository)

    def test_permissions_are_immutable_and_not_shell_commands(self) -> None:
        item = Permissions(("read",), ("references",), (), ("example.com",))
        self.assertEqual(Permissions.from_json(item.to_json()), item)
        for change in ({"tools": ["read"]}, {"tools": ("sh;curl",)},
                       {"read_paths": ("../private",)}, {"network_hosts": ("*",)}):
            with self.subTest(change=change), self.assertRaises(ContractError):
                replace(item, **change)

    def test_exact_compatibility_not_vram_only(self) -> None:
        profile = Compatibility("model", "a" * 40, "b" * 64, Backend.CUDA, 24082)
        device = GPUDevice("synthetic", Backend.CUDA, 24082)
        self.assertEqual(Compatibility.from_json(profile.to_json()), profile)
        self.assertEqual(Compatibility.from_json(Compatibility().to_json()), Compatibility())
        self.assertEqual(profile.reasons(shot(), device), ())
        self.assertIn("missing_device_profile", profile.reasons(shot(), None))
        self.assertIn("incompatible_hardware", profile.reasons(
            shot(), replace(device, backend=Backend.METAL, memory_mib=100000)))
        self.assertIn("incompatible_model_workflow", profile.reasons(
            replace(shot(), model_revision="c" * 40), device))
        with self.assertRaises(CatalogError):
            Compatibility("model")

    def test_shot_sidecar_preserves_v1_wire_contract(self) -> None:
        original = shot().to_json()
        binding = bind_shot(shot(), (SkillPin("camera", "r1", "c" * 64),))
        self.assertEqual(ShotSkills.from_json(binding.to_json()), binding)
        self.assertEqual(shot().to_json(), original)
        self.assertNotIn("skills", original)
        self.assertNotEqual(bind_shot(replace(shot(), prompt="changed"), binding.skills), binding)
        with self.assertRaises(ContractError):
            bind_shot(shot(), binding.skills * 2)
        with self.assertRaises(CatalogError):
            SkillPin("camera", "latest", "c" * 64)


if __name__ == "__main__":
    unittest.main()
