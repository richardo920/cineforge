"""Offline publication/cleanup faults; synthetic payloads and caller-owned leases."""

import json
import os
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

from test_comfy_run import WireFixture, plan_for, worker_record

from cineforge.comfy_policy import WorkerVerification, load_comfy_selection
from cineforge.comfy_run import ComfyRunError, execute_comfy
from cineforge.private_store import ArtifactStoreError, PrivateRunDirectory

ROOT = Path(__file__).resolve().parents[1]
CANARY = "synthetic-private-checkpoint-canary"


class PrivateCheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cineforge-checkpoint-recovery-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "run"

    def assert_static(self, error):
        self.assertIsInstance(error, ArtifactStoreError)
        self.assertNotIn(CANARY, "".join(traceback.format_exception(error)))
        self.assertIsNone(error.__context__)
        self.assertIsNone(error.__cause__)

    def test_first_publish_unlink_fault_keeps_identity_for_next_checkpoint(self):
        unlink = os.unlink
        calls = 0

        def unlink_once(name, **kwargs):
            nonlocal calls
            if str(name).startswith(".pending-"):
                calls += 1
                if calls == 1:
                    raise PermissionError(CANARY)
            return unlink(name, **kwargs)

        with PrivateRunDirectory(self.path) as store:
            with patch("cineforge.private_store.os.unlink", side_effect=unlink_once):
                with self.assertRaises(ArtifactStoreError) as caught:
                    store.checkpoint(b"first")
            self.assert_static(caught.exception)
            self.assertEqual((self.path / "run.json").read_bytes(), b"first")
            store.checkpoint(b"recovery")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovery")
            self.assertFalse(list(self.path.glob(".pending-*")))

    def test_post_replace_stat_fault_adopts_published_identity_before_failure(self):
        real_replace, real_stat = os.replace, os.stat
        replaced = False
        injected = False

        def replace(*args, **kwargs):
            nonlocal replaced
            result = real_replace(*args, **kwargs)
            replaced = True
            return result

        def stat(name, **kwargs):
            nonlocal injected
            if name == "run.json" and replaced and not injected:
                injected = True
                raise OSError(CANARY)
            return real_stat(name, **kwargs)

        with PrivateRunDirectory(self.path) as store:
            store.checkpoint(b"first")
            with patch("cineforge.private_store.os.replace", side_effect=replace), \
                    patch("cineforge.private_store.os.stat", side_effect=stat):
                with self.assertRaises(ArtifactStoreError) as caught:
                    store.checkpoint(b"second")
            self.assert_static(caught.exception)
            self.assertTrue(injected)
            self.assertEqual((self.path / "run.json").read_bytes(), b"second")
            store.checkpoint(b"recovery")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovery")

    def test_persistent_unlink_retains_one_owned_alias_then_recovers(self):
        with PrivateRunDirectory(self.path) as store:
            with patch("cineforge.private_store.os.unlink", side_effect=PermissionError(CANARY)):
                with self.assertRaises(ArtifactStoreError) as caught:
                    store.checkpoint(b"published")
                self.assert_static(caught.exception)
                pending = list(self.path.glob(".pending-*"))
                self.assertEqual(len(pending), 1)
                with self.assertRaises(ArtifactStoreError) as caught:
                    store.checkpoint(b"must not publish")
                self.assert_static(caught.exception)
                self.assertEqual(list(self.path.glob(".pending-*")), pending)
                self.assertEqual((self.path / "run.json").read_bytes(), b"published")
                self.assertEqual((self.path / "run.json").stat().st_nlink, 2)
            store.checkpoint(b"recovered")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovered")
            self.assertEqual((self.path / "run.json").stat().st_nlink, 1)
            self.assertFalse(list(self.path.glob(".pending-*")))

    def test_primary_link_and_finally_unlink_faults_are_static_without_adoption(self):
        with PrivateRunDirectory(self.path) as store:
            with patch("cineforge.private_store.os.link", side_effect=PermissionError(CANARY)), \
                    patch("cineforge.private_store.os.unlink", side_effect=PermissionError(CANARY)):
                with self.assertRaises(ArtifactStoreError) as caught:
                    store.checkpoint(b"unpublished")
            self.assert_static(caught.exception)
            self.assertFalse((self.path / "run.json").exists())
            self.assertIsNone(store._record_inode)
            self.assertEqual(len(list(self.path.glob(".pending-*"))), 1)
            store.checkpoint(b"recovered")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovered")

    def test_failed_identity_capture_retains_uncertainty_without_path_adoption(self):
        with PrivateRunDirectory(self.path) as store:
            with patch("cineforge.private_store.os.fstat", side_effect=OSError(CANARY)):
                with self.assertRaises(ArtifactStoreError) as caught:
                    store.checkpoint(b"unpublished")
            self.assert_static(caught.exception)
            self.assertIsNone(store._record_inode)
            self.assertFalse((self.path / "run.json").exists())
            pending, = self.path.glob(".pending-*")
            with self.assertRaises(ArtifactStoreError):
                store.checkpoint(b"must not adopt a path")
            self.assertEqual(pending.read_bytes(), b"")
            # Missing captured identity requires explicit operator removal, not
            # guessing that a later path observation is the original created file.
            pending.unlink()
            store.checkpoint(b"recovered")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovered")

    def test_persistent_directory_fsync_keeps_uncertainty_and_current_owned_identity(self):
        real_fsync = os.fsync
        with PrivateRunDirectory(self.path) as store:
            store.checkpoint(b"first")

            def fsync(descriptor):
                if descriptor == store._directory:
                    raise OSError(CANARY)
                return real_fsync(descriptor)

            with patch("cineforge.private_store.os.fsync", side_effect=fsync):
                for payload in (b"second", b"third"):
                    with self.assertRaises(ArtifactStoreError) as caught:
                        store.checkpoint(payload)
                    self.assert_static(caught.exception)
                    self.assertEqual((self.path / "run.json").read_bytes(), payload)
            store.checkpoint(b"recovery")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovery")

    def test_foreign_pending_aliases_are_neither_deleted_nor_accepted(self):
        for kind in ("regular", "symlink", "hardlink"):
            with self.subTest(kind=kind):
                path = self.root / kind
                outside = self.root / (kind + "-outside")
                outside.write_bytes(b"external bytes")
                with PrivateRunDirectory(path) as store:
                    with patch("cineforge.private_store.os.unlink", side_effect=PermissionError(CANARY)):
                        with self.assertRaises(ArtifactStoreError):
                            store.checkpoint(b"owned")
                    pending, = path.glob(".pending-*")
                    pending.unlink()
                    if kind == "regular":
                        pending.write_bytes(b"foreign temporary")
                    elif kind == "symlink":
                        pending.symlink_to(outside)
                    else:
                        os.link(outside, pending)
                    with self.assertRaises(ArtifactStoreError) as caught:
                        store.checkpoint(b"must not replace")
                    self.assert_static(caught.exception)
                    self.assertEqual(outside.read_bytes(), b"external bytes")
                    self.assertTrue(pending.exists())
                    self.assertEqual((path / "run.json").read_bytes(), b"owned")

    def test_unexpected_extra_temporary_link_blocks_cleanup_and_publication(self):
        real_link = os.link
        outside = self.root / "extra-alias"

        def link(source, destination, **kwargs):
            if destination == "run.json":
                real_link(source, outside, src_dir_fd=kwargs["src_dir_fd"],
                          follow_symlinks=False)
                raise OSError(CANARY)
            return real_link(source, destination, **kwargs)

        with PrivateRunDirectory(self.path) as store:
            with patch("cineforge.private_store.os.link", side_effect=link):
                with self.assertRaises(ArtifactStoreError):
                    store.checkpoint(b"owned temporary")
            self.assertFalse((self.path / "run.json").exists())
            self.assertEqual(len(list(self.path.glob(".pending-*"))), 1)
            with self.assertRaises(ArtifactStoreError):
                store.checkpoint(b"must not publish")
            self.assertEqual(outside.read_bytes(), b"owned temporary")
            outside.unlink()
            store.checkpoint(b"recovered")
            self.assertEqual((self.path / "run.json").read_bytes(), b"recovered")

    def test_foreign_record_inode_stays_rejected_after_publication_failure(self):
        real_fsync = os.fsync
        outside = self.root / "foreign"
        outside.write_bytes(b"external record")
        with PrivateRunDirectory(self.path) as store:
            def fsync(descriptor):
                if descriptor == store._directory:
                    raise OSError(CANARY)
                return real_fsync(descriptor)

            with patch("cineforge.private_store.os.fsync", side_effect=fsync):
                with self.assertRaises(ArtifactStoreError):
                    store.checkpoint(b"owned")
            os.replace(outside, self.path / "run.json")
            with self.assertRaises(ArtifactStoreError) as caught:
                store.checkpoint(b"must not overwrite")
            self.assert_static(caught.exception)
            self.assertEqual((self.path / "run.json").read_bytes(), b"external record")

    def test_submitted_job_checkpoint_fault_recovers_metadata_and_bounded_cleanup(self):
        from test_cloud_budget import BudgetTests

        fixture = BudgetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        selection = load_comfy_selection(ROOT / "config/comfy/wan21-t2v-1.3b.json")
        wire = WireFixture(selection, mode="running")
        real_replace, real_stat = os.replace, os.stat
        replacements = 0
        injected = False

        def replace(source, destination, **kwargs):
            nonlocal replacements
            result = real_replace(source, destination, **kwargs)
            if destination == "run.json":
                replacements += 1
            return result

        def stat(name, **kwargs):
            nonlocal injected
            if name == "run.json" and replacements == 2 and not injected:
                injected = True
                raise OSError(CANARY)
            return real_stat(name, **kwargs)

        with fixture.session() as budget:
            provider = fixture.provider(budget)
            fixture.acquire(provider)
            fixture.clock.sleep(1735)
            budget.heartbeat("first")
            cap = budget.remaining_seconds("first")
            started = fixture.clock.monotonic()
            with patch("cineforge.private_store.os.replace", side_effect=replace), \
                    patch("cineforge.private_store.os.stat", side_effect=stat):
                with self.assertRaises(ComfyRunError):
                    execute_comfy(plan_for(selection), selection,
                        WorkerVerification.from_dict(worker_record(selection)), self.path,
                        allow_live=True, transport=wire, timeout_seconds=cap,
                        clock=fixture.clock.monotonic, sleep=fixture.clock.sleep)
            self.assertTrue(injected)
            record = json.loads((self.path / "run.json").read_bytes())
            self.assertEqual(record["outcome"], "incomplete")
            self.assertTrue(record["known_run"]["job_id"])
            self.assertEqual(record["known_run"]["status"], "cancelled")
            self.assertEqual(record["cancellation"],
                             {"attempted": True, "confirmed_terminal": True})
            self.assertEqual(record["request"], selection.request(plan_for(selection).shots[0]).to_dict())
            self.assertIsNone(record["server_output"])
            self.assertLessEqual(fixture.clock.monotonic() - started, cap)
            self.assertTrue(all(c[3] <= cap for c in wire.calls))
            self.assertGreaterEqual(1800 - (fixture.clock.time() - 10000), 60)
            provider.release("pod_1")
            self.assertEqual(budget.snapshots[0].state, "released")
            self.assertFalse(fixture.cloud.pods)

    def test_succeeded_output_checkpoint_fault_retains_descriptor_without_redownload(self):
        selection = load_comfy_selection(ROOT / "config/comfy/wan21-t2v-1.3b.json")
        wire = WireFixture(selection)
        real_replace, real_stat = os.replace, os.stat
        replacements = 0
        injected = False

        def replace(source, destination, **kwargs):
            nonlocal replacements
            result = real_replace(source, destination, **kwargs)
            if destination == "run.json":
                replacements += 1
            return result

        def stat(name, **kwargs):
            nonlocal injected
            if name == "run.json" and replacements == 3 and not injected:
                injected = True
                raise OSError(CANARY)
            return real_stat(name, **kwargs)

        with patch("cineforge.private_store.os.replace", side_effect=replace), \
                patch("cineforge.private_store.os.stat", side_effect=stat):
            with self.assertRaises(ComfyRunError):
                execute_comfy(plan_for(selection), selection,
                    WorkerVerification.from_dict(worker_record(selection)), self.path,
                    allow_live=True, transport=wire, clock=lambda: 0)
        self.assertTrue(injected)
        record = json.loads((self.path / "run.json").read_bytes())
        self.assertEqual(record["outcome"], "incomplete")
        self.assertEqual(record["known_run"]["status"], "succeeded")
        self.assertEqual(record["server_output"],
                         {"filename": "fixture.webm", "subfolder": "", "type": "output"})
        self.assertFalse(record["cancellation"]["attempted"])
        self.assertNotIn("manifest", record)
        self.assertEqual(sum(c[1] == "/prompt" for c in wire.calls), 1)
        self.assertFalse(any(c[1].startswith("/view?") for c in wire.calls))


if __name__ == "__main__":
    unittest.main()
