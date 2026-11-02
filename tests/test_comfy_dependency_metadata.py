"""Synthetic dependency-header multisets; no real wheel/SDK installation or import."""

import copy
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_comfy_runtime import plan, wheel

from cineforge import comfy_runtime_worker as worker
from cineforge.comfy_runtime_plan import COMFY_REQUIRES_DIST_MAX, _wheel, comfy_plan, comfy_subject


def dependency_wheel(headers):
    metadata = (
        "Metadata-Version: 2.1\nName: fixture\nVersion: 1\n"
        + "".join("Requires-Dist: " + value + "\n" for value in headers)
    ).encode()
    raw, declaration, bodies = wheel(changes={"fixture-1.dist-info/METADATA": metadata})
    declaration["requires_dist"] = sorted(headers)
    return raw, declaration, bodies


class DependencyMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cf-dependency-metadata-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o700)
        self.target = self.root / "target"
        self.target.mkdir(mode=0o700)
        self.path = self.root / "fixture.whl"

    def install(self, raw, declaration, manifest):
        self.path.write_bytes(raw)
        self.path.chmod(0o600)
        worker.install_wheel(self.path, self.target, declaration, manifest, time.monotonic() + 5)

    def test_sorted_repetitions_are_preserved_and_change_the_review_subject(self):
        declaration = plan()
        headers = ["synthetic-a>=1", "synthetic-a>=1", "synthetic-b>=2"]
        declaration["wheels"][0]["requires_dist"] = headers[:]
        accepted = comfy_plan(copy.deepcopy(declaration))
        self.assertEqual(accepted["wheels"][0]["requires_dist"], headers)
        original = comfy_subject(declaration)
        declaration["wheels"][0]["requires_dist"].pop(0)
        self.assertNotEqual(comfy_subject(declaration), original)
        declaration["wheels"][0]["requires_dist"] = list(reversed(headers))
        with self.assertRaises(ValueError):
            comfy_plan(declaration)

    def test_actual_like_header_counts_accept_exact_multiplicity_and_unsorted_raw_headers(self):
        for raw_count, unique_count in ((131, 128), (457, 415)):
            with self.subTest(raw_count=raw_count):
                self.target = self.root / f"target-{raw_count}"
                self.target.mkdir(mode=0o700)
                headers = [f"synthetic-dependency-{i:04d}>=1" for i in range(unique_count)]
                headers += headers[: raw_count - unique_count]
                raw, declaration, bodies = dependency_wheel(list(reversed(headers)))
                self.assertEqual(len(declaration["requires_dist"]), raw_count)
                self.assertEqual(len(set(declaration["requires_dist"])), unique_count)
                manifest = {}
                self.install(raw, declaration, manifest)
                self.assertEqual(
                    manifest,
                    {
                        "site/" + path: (len(body), hashlib.sha256(body).hexdigest())
                        for path, body in bodies.items()
                    },
                )
                self.assertEqual(
                    (self.target / "site/fixture-1.dist-info/METADATA").read_bytes(),
                    bodies["fixture-1.dist-info/METADATA"],
                )

    def test_removed_added_changed_or_reordered_declarations_fail_before_extraction(self):
        raw, declaration, _ = dependency_wheel(
            ["synthetic-b>=1", "synthetic-a>=1", "synthetic-a>=1"]
        )
        variants = (
            ["synthetic-a>=1", "synthetic-b>=1"],
            ["synthetic-a>=1"] * 3 + ["synthetic-b>=1"],
            ["synthetic-a>=2", "synthetic-a>=2", "synthetic-b>=1"],
            list(reversed(declaration["requires_dist"])),
        )
        for headers in variants:
            with self.subTest(headers=headers):
                changed = copy.deepcopy(declaration)
                changed["requires_dist"] = headers
                manifest = {}
                with self.assertRaises(ValueError):
                    self.install(raw, changed, manifest)
                self.assertEqual(manifest, {})
                self.assertEqual(list(self.target.iterdir()), [])

    def test_count_and_entry_bounds_apply_to_both_controller_and_worker(self):
        headers = ["synthetic-dependency>=1"] * COMFY_REQUIRES_DIST_MAX
        raw, declaration, _ = dependency_wheel(headers)
        _wheel(declaration)
        manifest = {}
        self.install(raw, declaration, manifest)
        self.assertTrue(manifest)
        for invalid in (headers + headers[:1], ["x" * 4097], [None], [True], ("tuple",), "text"):
            changed = copy.deepcopy(declaration)
            changed["requires_dist"] = invalid
            with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                _wheel(changed)
            with (
                patch.object(worker, "verify_binary") as read,
                patch.object(worker, "_install_wheel") as extract,
                self.assertRaises(ValueError),
            ):
                worker.install_wheel(self.path, self.target, changed, {}, time.monotonic() + 5)
            read.assert_not_called()
            extract.assert_not_called()

    def test_header_order_remains_bound_by_the_exact_metadata_byte_hash(self):
        headers = ["synthetic-a>=1", "synthetic-a>=1", "synthetic-b>=1"]
        _, original, _ = dependency_wheel(headers)
        raw, changed, _ = dependency_wheel(list(reversed(headers)))
        self.assertEqual(changed["requires_dist"], original["requires_dist"])
        self.assertNotEqual(changed["metadata_sha256"], original["metadata_sha256"])
        changed["metadata_sha256"] = original["metadata_sha256"]
        with self.assertRaises(ValueError):
            self.install(raw, changed, {})
        self.assertEqual(list(self.target.iterdir()), [])
