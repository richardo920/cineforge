"""Fixed inert native-file measurements; no SDK/import/install/cloud effects."""

import contextlib
import copy
import hashlib
import os
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from test_native_remote import RemoteFixture, recipe

from cineforge import native_remote as controller
from cineforge import native_remote_bootstrap as bootstrap
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_remote_plan import nr_digest


class FixedFileMeasurementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cf-fixed-native-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o700)
        self.body = self.root / "fixed-native-body"
        self.body.write_bytes(b"inert synthetic bytes, never executed or imported")
        self.body.chmod(0o600)
        self.alias = self.root / "fixed-alias"
        self.alias.symlink_to(self.body.name)
        for item in (
            patch.object(bootstrap, "NR_CAP_ROOTS", (str(self.root) + "/",)),
            patch.object(
                bootstrap,
                "NR_CAP_ALIASES",
                (("cuda_driver", str(self.root / "missing")), ("interpreter", str(self.alias))),
            ),
            patch.object(bootstrap.sys, "executable", str(self.alias)),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.deadline = time.monotonic() + 10

    def test_protected_body_alias_and_missing_are_measurements_without_execution(self):
        with (
            patch("subprocess.Popen", side_effect=AssertionError("child")),
            patch("importlib.import_module", side_effect=AssertionError("import")),
        ):
            result = bootstrap.nr_capabilities(self.deadline, {"fixture": "binding"})
        self.assertEqual(result["interpreter_path"], str(self.body))
        self.assertEqual(result["files"][0]["status"], "missing")
        self.assertEqual(
            result["files"][1]["sha256"], hashlib.sha256(self.body.read_bytes()).hexdigest()
        )
        self.assertEqual(result["files"][1]["path"], str(self.body))
        self.assertFalse({"ready", "approved", "runtime", "abi"} & set(result))

    def test_protected_roots_links_ownership_modes_and_hardlinks_refuse(self):
        cases = ("escape", "mode", "hardlink", "loop", "ancestor")
        for case in cases:
            with self.subTest(case=case):
                if self.alias.exists() or self.alias.is_symlink():
                    self.alias.unlink()
                self.alias.symlink_to(self.body.name)
                self.body.chmod(0o600)
                if case == "escape":
                    self.alias.unlink()
                    self.alias.symlink_to(Path(__file__).resolve())
                elif case == "mode":
                    self.body.chmod(0o620)
                elif case == "hardlink":
                    os.link(self.body, self.root / "hardlink")
                elif case == "loop":
                    self.alias.unlink()
                    self.alias.symlink_to(self.alias.name)
                else:
                    self.root.chmod(0o770)
                try:
                    with self.assertRaises((ValueError, OSError)):
                        bootstrap.nr_cap_file(str(self.alias), self.deadline)
                finally:
                    self.root.chmod(0o700)
                    if (self.root / "hardlink").exists():
                        (self.root / "hardlink").unlink()

    def test_changed_body_and_retargeted_alias_refuse_after_actual_read(self):
        read = os.read
        for alias_change in (False, True):
            self.alias.unlink()
            self.alias.symlink_to(self.body.name)
            self.body.write_bytes(b"original synthetic body")
            changed = False

            def mutate(fd, count):
                nonlocal changed
                result = read(fd, count)
                if result and not changed:
                    changed = True
                    if alias_change:
                        other = self.root / "other"
                        other.write_bytes(b"other inert bytes")
                        other.chmod(0o600)
                        self.alias.unlink()
                        self.alias.symlink_to(other.name)
                    else:
                        self.body.write_bytes(b"changed synthetic body is longer")
                return result

            with patch.object(bootstrap.os, "read", mutate), self.assertRaises(ValueError):
                bootstrap.nr_cap_file(str(self.alias), self.deadline)
            self.assertTrue(changed)

    def test_expired_or_slow_read_never_returns_measurement(self):
        with patch.object(bootstrap.os, "open", side_effect=AssertionError("open")):
            with self.assertRaises(ValueError):
                bootstrap.nr_cap_file(str(self.alias), time.monotonic() - 1)
        read = os.read
        now = [time.monotonic()]
        cap = now[0] + 3

        def late(fd, count):
            result = read(fd, count)
            now[0] = cap + 1
            return result

        with (
            patch.object(bootstrap.time, "monotonic", lambda: now[0]),
            patch.object(bootstrap.os, "read", late),
            self.assertRaises(ValueError),
        ):
            bootstrap.nr_cap_file(str(self.alias), cap)


class CapabilityChannelTests(RemoteFixture):
    def setUp(self):
        super().setUp()
        self.measurements = 0
        self.change_facts = False
        for item in (
            patch.object(bootstrap, "nr_capabilities", self.facts),
            patch.object(controller, "NR_CAP_ROOTS", (str(self.root) + "/",)),
        ):
            item.start()
            self.addCleanup(item.stop)

    def facts(self, deadline, binding):
        self.measurements += 1
        rows = []
        for role, alias in bootstrap.NR_CAP_ALIASES:
            row = dict(role=role, alias=alias, status="missing")
            if role in ("interpreter", "seccomp"):
                row.update(asdict(getattr(self.selected, role)), status="measured")
            rows.append(row)
        if self.change_facts and self.measurements > 1:
            next(row for row in rows if row["role"] == "interpreter")["sha256"] = "b" * 64
        return dict(
            version=1,
            policy=bootstrap.NR_CAP_POLICY,
            binding_sha256=nr_digest(binding),
            interpreter_path=self.selected.interpreter.path,
            files=rows,
        )

    def test_capability_first_then_stage_reuses_one_root_binding_and_original_caps(self):
        cap = time.monotonic() + 180
        measured = self.bridge.measure_capabilities(absolute_deadline=cap)
        self.assertEqual([r["operation"] for r in self.requests], ["native_capabilities"])
        self.assertEqual(
            set(p.name for p in self.remote.iterdir()),
            {
                "operation.lock",
                "binding.json",
                "local-cap.json",
                "capabilities-consumed",
                "capabilities.json",
            },
        )
        self.assertEqual(self.child_count, 0)
        deadline, expiry = self.bridge._deadline, self.bridge._expires_unix
        saved = bootstrap.nr_read(self.remote / "local-cap.json")
        self.remaining = 2000
        installed = self.bridge.stage(recipe(), b"x")
        self.assertEqual(installed["interpreter"], asdict(self.selected.interpreter))
        self.assertEqual((self.bridge._deadline, self.bridge._expires_unix), (deadline, expiry))
        self.assertEqual(bootstrap.nr_read(self.remote / "local-cap.json"), saved)
        self.assertTrue(all(deadline <= cap for deadline in self.deadlines))
        preflight = next(r for r in self.requests if r["operation"] == "native_preflight")
        self.assertEqual(preflight["capabilities_sha256"], nr_digest(measured))
        self.assertEqual(self.measurements, 2)
        self.assertEqual(self.child_count, 0)

    def test_direct_stage_keeps_existing_preflight_without_capability_rpc(self):
        self.bridge.stage(recipe(), b"x")
        self.assertEqual(self.requests[0]["operation"], "native_preflight")
        self.assertNotIn("capabilities_sha256", self.requests[0])
        self.assertEqual(self.measurements, 0)
        self.assertTrue((self.remote / "preflight-consumed").exists())

    def test_lost_measurement_ack_is_consumed_and_cannot_retry_or_stage(self):
        self.lost = "native_capabilities"
        self.static_failure(self.bridge.measure_capabilities)
        self.assertTrue((self.remote / "capabilities-consumed").exists())
        before = len(self.requests)
        self.static_failure(self.bridge.measure_capabilities)
        self.static_failure(lambda: self.bridge.stage(recipe(), b"x"))
        self.assertEqual(len(self.requests), before)
        self.assertFalse((self.remote / "package.blob").exists())

    def test_changed_captured_native_file_refuses_before_preflight_or_download(self):
        self.bridge.measure_capabilities()
        self.change_facts = True
        self.static_failure(lambda: self.bridge.stage(recipe(), b"x"))
        self.assertFalse((self.remote / "preflight.json").exists())
        self.assertFalse((self.remote / "package.blob").exists())
        self.assertTrue((self.remote / "failed").exists())

    def test_invalid_deadlines_make_no_rpc_or_measurement(self):
        for cap in (False, float("nan"), float("inf"), "later", time.monotonic() - 1):
            with self.subTest(cap=cap):
                self.static_failure(lambda: self.bridge.measure_capabilities(absolute_deadline=cap))
        self.assertEqual(self.requests, [])
        self.assertEqual(self.measurements, 0)

    def test_source_revocation_after_measurement_refuses_final_publication(self):
        self.mutate = lambda request, result: (
            setattr(self.worker, "_source", b"changed") or result
        )
        self.static_failure(self.bridge.measure_capabilities)
        self.assertTrue(self.bridge._unresolved)
        self.assertIsNone(self.bridge._capabilities_sha256)
        self.assertEqual(self.child_count, 0)

    def test_late_actual_worker_lock_finalizer_never_acknowledges_measurements(self):
        original = bootstrap.nr_lock
        now = [time.monotonic()]

        @contextlib.contextmanager
        def late(root):
            with original(root):
                yield
            now[0] += 10000

        with (
            patch.object(bootstrap, "nr_lock", late),
            patch.object(bootstrap.time, "monotonic", lambda: now[0]),
        ):
            self.static_failure(self.bridge.measure_capabilities)
        self.assertTrue((self.remote / "failed").exists())
        self.assertIsNone(self.bridge._capabilities_sha256)

    def test_controller_lock_release_and_final_validation_remain_inside_original_cap(self):
        now = [time.monotonic()]
        original = self.bridge._lock

        class LateLock:
            def acquire(self, **kwargs):
                return original.acquire(**kwargs)

            def release(self):
                original.release()
                now[0] += 10000

        self.bridge._lock = LateLock()
        with patch.object(controller.time, "monotonic", lambda: now[0]):
            self.static_failure(self.bridge.measure_capabilities)
        self.assertIsNone(self.bridge._capabilities_sha256)

    def test_malformed_and_path_selecting_facts_refuse_without_package_effects(self):
        variants = (
            lambda r: r.update(version=True),
            lambda r: r.update(binding_sha256="b" * 64),
            lambda r: r.update(approved=True),
            lambda r: r["files"].append(copy.deepcopy(r["files"][0])),
            lambda r: r["files"][0].update(alias="/arbitrary/path"),
            lambda r: next(p for p in r["files"] if p["status"] == "measured").update(
                size_bytes=True
            ),
            lambda r: next(p for p in r["files"] if p["status"] == "measured").update(
                path="/outside/selection"
            ),
        )
        for index, mutate in enumerate(variants):
            with self.subTest(index=index):
                self.remote = self.root / f"remote-invalid-{index}"
                self.bridge = RemoteNativeSpeechRuntime(self.worker, allow_native=True)
                self.mutate = lambda request, result: (mutate(result) or result)
                self.static_failure(self.bridge.measure_capabilities)
                self.assertTrue(self.bridge._unresolved)
                self.assertFalse((self.remote / "package.blob").exists())
        self.mutate = None
        request = copy.deepcopy(self.requests[0])
        request.update(
            profile=self.worker._profile,
            guard_sha256=request.get("guard_sha256", "a" * 64),
            path="/not/a/selector",
        )
        with self.assertRaisesRegex(ValueError, "no path or command"):
            bootstrap.nr_dispatch(request, time.monotonic() + 5)

    def test_expired_capability_channel_cannot_be_renewed_by_stage(self):
        self.bridge.measure_capabilities()
        deadline = self.bridge._deadline
        with patch.object(controller.time, "monotonic", lambda: deadline + 1):
            before = len(self.requests)
            self.static_failure(lambda: self.bridge.stage(recipe(), b"x"))
            self.assertEqual(len(self.requests), before)
        self.assertFalse((self.remote / "package.blob").exists())
