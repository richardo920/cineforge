"""Offline source provenance, review and immutable shot binding acceptance."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from cineforge.contracts import ContractError, Shot
from cineforge.devices import Backend, GPUDevice, Precision
from cineforge.rights import EvidenceRef, RightsLedger, RightsRecord, RightsRef, Use, with_review
from cineforge.skills import (
    CatalogEntry,
    CatalogError,
    Compatibility,
    Permissions,
    ShotSkills,
    SkillCatalog,
    SkillPin,
    SourcePin,
    SourceReview,
    bind_shot,
)


def shot() -> Shot:
    return Shot("shot", "scene", 1, "synthetic", 42, 2.0625, "model", "a" * 40, "b" * 64)


def reviewed_entry() -> CatalogEntry:
    source = SourcePin("https://github.com/example/creative", "a" * 40, "SKILL.md", "b" * 64)
    entry = CatalogEntry("camera", "r1", "camera", "synthetic source", source, source.path,
                         Permissions(), Compatibility(), RightsRef("camera-rights", "r1"))
    return replace(entry, review=SourceReview(entry.digest, "reviewer", "urn:review:one", "c" * 64))


def ledger_for(entry: CatalogEntry, **changes: object) -> RightsLedger:
    record = RightsRecord(entry.rights, entry.subject,
                          source=EvidenceRef(entry.source.uri, entry.source.sha256),
                          source_revision=entry.source.revision, license_id="LicenseRef-Synthetic",
                          terms=EvidenceRef("urn:terms:synthetic", "e" * 64),
                          notices=("Synthetic notice",), attribution="Synthetic source",
                          allowed_uses=(Use("reference"), Use("execute")))
    record = with_review(record, reviewer="reviewer", evidence=EvidenceRef("urn:review:one", "f"*64))
    ledger = RightsLedger()
    ledger.register(replace(record, **changes))
    return ledger


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
        credential_url = "".join(("https://", "synthetic-user", ":", "synthetic-pass", "@github.com/a/b"))
        for repository in ("http://github.com/a/b", credential_url,
                           "https://github.com/a/b?token=value", "https://github.com/a/b/",
                           "https://github.com/a/\nb"):
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


class ReadinessTests(unittest.TestCase):
    def test_reviewed_reference_roundtrip_never_authorizes_execution(self) -> None:
        entry = reviewed_entry()
        self.assertEqual(CatalogEntry.from_json(entry.to_json()), entry)
        catalog = SkillCatalog((entry,))
        result = catalog.reference_readiness(entry.pin, shot=shot(), ledger=ledger_for(entry))
        self.assertTrue(result.ready)
        self.assertFalse(result.execution_authorized)
        self.assertEqual(len(result.attributions), 1)
        result = catalog.reference_readiness(entry.pin, shot=shot(), ledger=ledger_for(entry),
                                             use=Use("execute"))
        self.assertFalse(result.ready)
        self.assertIn("reference_use_required", result.reasons)

    def test_unknown_missing_unreviewed_and_revoked_rights_fail_closed(self) -> None:
        entry = reviewed_entry()
        catalog = SkillCatalog((entry,))
        for changes, reason in (({"license_id": "unknown"}, "unknown_terms"),
                                ({"notices": ()}, "missing_notices"),
                                ({"review": None}, "unreviewed_rights"),
                                ({"allowed_uses": (Use("execute"),)}, "use_not_permitted")):
            with self.subTest(changes=changes):
                result = catalog.reference_readiness(entry.pin, shot=shot(),
                                                      ledger=ledger_for(entry, **changes))
                self.assertFalse(result.ready)
                self.assertIn(reason, result.reasons)
                self.assertEqual(result.attributions, ())
        result = catalog.reference_readiness(entry.pin, shot=shot(), ledger=RightsLedger())
        self.assertIn("missing_rights", result.reasons)
        ledger = ledger_for(entry)
        binding = bind_shot(shot(), (entry.pin,))
        self.assertTrue(catalog.resolve_shot(binding, shot=shot(), ledger=ledger)[0].ready)
        ledger.revoke(entry.rights, reason="synthetic revocation")
        result = catalog.resolve_shot(binding, shot=shot(), ledger=ledger)[0]
        self.assertFalse(result.ready)
        self.assertIn("revoked_rights", result.reasons)

    def test_source_content_permissions_and_profiles_invalidate_review(self) -> None:
        entry = reviewed_entry()
        for changes in ({"source": replace(entry.source, sha256="d" * 64)},
                        {"permissions": Permissions(("read",))},
                        {"compatibility": Compatibility("model", "a" * 40, "b" * 64)},
                        {"rights": RightsRef("replacement", "r1")}):
            with self.subTest(changes=changes):
                changed = replace(entry, **changes)
                result = SkillCatalog((changed,)).reference_readiness(
                    changed.pin, shot=shot(), ledger=ledger_for(changed))
                self.assertFalse(result.ready)
                self.assertIn("stale_source_review", result.reasons)
        unreviewed = replace(entry, review=None)
        result = SkillCatalog((unreviewed,)).reference_readiness(
            unreviewed.pin, shot=shot(), ledger=ledger_for(unreviewed))
        self.assertIn("unreviewed_source", result.reasons)

    def test_pin_resolution_never_upgrades_or_overwrites_revisions(self) -> None:
        entry = reviewed_entry()
        catalog = SkillCatalog((entry,))
        with self.assertRaises(CatalogError):
            catalog.register(replace(entry, summary="overwrite"))
        with self.assertRaises(CatalogError):
            catalog.lookup(replace(entry.pin, sha256="d" * 64))
        with self.assertRaises(CatalogError):
            catalog.lookup(replace(entry.pin, revision="absent"))
        upgraded = replace(entry, revision="r2", source=replace(entry.source, revision="c" * 40))
        catalog.register(upgraded)
        self.assertEqual(catalog.lookup(entry.pin), entry)
        self.assertEqual(len(catalog.pins), 2)
        self.assertIn("stale_source_review", catalog.reference_readiness(
            upgraded.pin, shot=shot(), ledger=ledger_for(upgraded)).reasons)

    def test_changed_shot_invalidates_sidecar_including_character(self) -> None:
        entry = reviewed_entry()
        binding = bind_shot(shot(), (entry.pin,))
        catalog = SkillCatalog((entry,))
        for changes in ({"prompt": "changed"}, {"model_revision": "d" * 40},
                        {"character_revision": "character@r2"}, {"revision": 2}):
            with self.subTest(changes=changes), self.assertRaises(CatalogError):
                catalog.resolve_shot(binding, shot=replace(shot(), **changes),
                                     ledger=ledger_for(entry))

    def test_incompatible_recipe_cannot_be_ready(self) -> None:
        base = replace(reviewed_entry(), compatibility=Compatibility(
            "model", "a" * 40, "b" * 64, Backend.CUDA, 24082, Precision.FP16))
        entry = replace(base, review=replace(base.review, declaration_sha256=base.digest))
        catalog = SkillCatalog((entry,))
        for device, reason in ((None, "missing_device_profile"),
                               (GPUDevice("cpu", Backend.CPU, 100000), "incompatible_hardware"),
                               (GPUDevice("low", Backend.CUDA, 24081), "incompatible_hardware"),
                               (GPUDevice("wrong-precision", Backend.CUDA, 24082),
                                "incompatible_hardware")):
            result = catalog.reference_readiness(entry.pin, shot=shot(), ledger=ledger_for(entry),
                                                  device=device)
            self.assertFalse(result.ready)
            self.assertIn(reason, result.reasons)


class CuratedTests(unittest.TestCase):
    def test_curated_references_have_traceable_restricted_reference_rights(self) -> None:
        from cineforge.curated_skills import curated_references

        catalog, ledger = curated_references()
        self.assertEqual(len(catalog.pins), 6)
        self.assertEqual({catalog.lookup(pin).category for pin in catalog.pins},
                         {"story", "camera", "audio", "editing", "model_card", "recipe"})
        for pin in catalog.pins:
            entry = catalog.lookup(pin)
            self.assertEqual(CatalogEntry.from_json(entry.to_json()), entry)
            context = replace(shot(), model_id=entry.compatibility.model_id or "model",
                              model_revision=entry.compatibility.model_revision or "a" * 40,
                              workflow_digest=entry.compatibility.workflow_digest or "b" * 64)
            result = catalog.reference_readiness(pin, shot=context, ledger=ledger,
                                                  device=GPUDevice("synthetic",Backend.CUDA,24082, frozenset({Precision.FP16})))
            self.assertTrue(result.ready, (pin, result.reasons))
            self.assertFalse(result.execution_authorized)
            self.assertEqual(entry.permissions, Permissions())
            self.assertFalse(ledger.preflight_use(entry.subject, entry.rights, Use("execute")).allowed)
            self.assertFalse(ledger.preflight_use(entry.subject, entry.rights,
                                                  Use("reference", commercial=True)).allowed)
        recipe = next(catalog.lookup(pin) for pin in catalog.pins if pin.entry_id=="wan21-comfy-recipe")
        self.assertEqual({profile.origin for profile in recipe.profiles},
                         {"upstream_recommendation", "local_observation"})

    def test_loading_changed_curated_content_does_not_generate_fresh_review(self) -> None:
        import cineforge.curated_skills as curated

        sources = curated._SOURCES
        with patch.object(curated, "_SOURCES", (replace(sources[0], sha256="0" * 64), *sources[1:])):
            catalog, ledger = curated.curated_references()
            entry = next(catalog.lookup(pin) for pin in catalog.pins if pin.entry_id=="write-screenplay")
            result = catalog.reference_readiness(entry.pin, shot=shot(), ledger=ledger)
            self.assertFalse(result.ready)
            self.assertIn("stale_source_review", result.reasons)
            self.assertIn("stale_review", result.reasons)


if __name__ == "__main__":
    unittest.main()
