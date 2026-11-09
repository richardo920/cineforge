"""Synthetic mapped-namespace/read-only-mount probes; no actual GPU or provider."""
from __future__ import annotations

import copy
import hashlib
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cineforge import native_driver as driver
from cineforge.native_speech_integrity import NativeBinary, NativeSpeechError, verify_binary

BINDING = "a" * 64


class DriverMountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.body = self.root / "nvidia-smi"
        self.body.write_bytes(b"synthetic driver body, not executable")
        self.body.chmod(0o600)
        self.deadline = time.monotonic()+10
        self.uid = 65534
        self.mount_id = 17
        self.mapping = "0 100000 65536\n"
        self.mount_flags = "ro,relatime"
        self.fs = "ext4"
        self.point = str(self.body)
        self.readonly = True
        self.directory_stats = dict(st_nlink=2, st_size=4096, st_mtime_ns=0, st_ctime_ns=0)
        self.real_stat, self.real_fstat, self.real_lstat = os.stat, os.fstat, os.lstat
        self.identity = self.real_stat(self.body)
        # Explicit Linux worker wire identity, including on a Darwin controller.
        self.major_minor = "8:1"
        self.extra_mount = ""
        self.descriptor_mount = 17
        self.context = []
        for context in (
            patch.dict(driver.DRIVER_ALIASES, {str(self.body): "nvidia_smi"}),
            patch.object(driver, "DRIVER_ROOTS", (str(self.root)+"/",)),
            patch.object(driver.os, "getuid", return_value=0),
            patch.object(driver.os, "major", side_effect=lambda device: 8 if device == 2049 else self.fail("unexpected synthetic Linux device")),
            patch.object(driver.os, "minor", side_effect=lambda device: 1 if device == 2049 else self.fail("unexpected synthetic Linux device")),
            patch.object(driver.os, "stat", side_effect=self.fake_stat),
            patch.object(driver.os, "fstat", side_effect=self.fake_fstat),
            patch.object(driver.os, "lstat", side_effect=lambda *a, **kw: self.changed(self.real_lstat(*a, **kw))),
            patch.object(driver.os, "fstatvfs", side_effect=lambda fd: SimpleNamespace(f_flag=os.ST_RDONLY if self.readonly else 0)),
            patch.object(driver, "_proc", side_effect=self.proc),
        ):
            context.start()
            self.addCleanup(context.stop)
        original_readlink = os.readlink
        self.namespaces = {"/proc/self/ns/user": "user:[12]", "/proc/self/ns/mnt": "mnt:[13]"}
        context = patch.object(driver.os, "readlink", side_effect=lambda path, **kw:
            self.namespaces[str(path)] if str(path) in self.namespaces else original_readlink(path, **kw))
        context.start()
        self.addCleanup(context.stop)

    def changed(self, info):
        values = {key: getattr(info, key) for key in dir(info) if key.startswith("st_")}
        values["st_uid"] = self.uid if (info.st_dev, info.st_ino) == (self.identity.st_dev, self.identity.st_ino) else 0
        # Real descriptors/inodes and mutation checks remain in use. Only the
        # device ABI is synthetic Linux, rather than leaking the test host's dev_t.
        values["st_dev"] = 2049
        if stat.S_ISDIR(info.st_mode):
            # Synthetic protected ancestors must not inherit unrelated /tmp churn
            # from concurrent test processes. Keep actual inode/owner/mode, which
            # the explicit replacement/mode race tests still mutate and verify.
            values.update(self.directory_stats)
        return SimpleNamespace(**values)

    def fake_stat(self, *args, **kwargs):
        return self.changed(self.real_stat(*args, **kwargs))

    def fake_fstat(self, fd):
        return self.changed(self.real_fstat(fd))

    def proc(self, path, limit, deadline):
        if path == "/proc/self/uid_map":
            return self.mapping
        if path.startswith("/proc/self/fdinfo/"):
            return "mnt_id:\t"+str(self.descriptor_mount)+"\n"
        self.assertEqual(path, "/proc/self/mountinfo")
        return f"{self.mount_id} 1 {self.major_minor} / {self.point} {self.mount_flags} - {self.fs} /host/redacted rw\n"+self.extra_mount

    def measure(self):
        return driver.read_native_file(str(self.body), self.deadline, (str(self.root)+"/",), binding_sha256=BINDING)

    def test_actual_body_hash_receipt_and_later_reverification(self):
        binary = self.measure()
        self.assertEqual(binary["sha256"], hashlib.sha256(self.body.read_bytes()).hexdigest())
        self.assertEqual(binary["driver_mount"]["identity"][2], 65534)
        driver.validate_driver_receipt(binary["driver_mount"], path=binary["path"],
            size_bytes=binary["size_bytes"], sha256=binary["sha256"], binding_sha256=BINDING, role="nvidia_smi", alias=str(self.body))
        driver.verify_driver_binary(binary, self.deadline, BINDING)
        with self.assertRaises((NativeSpeechError, ValueError)):
            verify_binary(NativeBinary(**{k: binary[k] for k in ("path", "size_bytes", "sha256")}), self.deadline)
        with self.assertRaises(TypeError):
            NativeBinary(**binary)

    def test_uid_alone_unbound_namespace_and_non_file_mount_refuse(self):
        for field, values in (
            ("uid", [1, 65533, 65535]),
            ("mapping", ["0 0 4294967295\n", "0 100000 65536\n65536 0 1\n", "", "invalid\n"]),
            ("readonly", [False]), ("mount_flags", ["rw,relatime"]),
            ("fs", ["overlay", "tmpfs"]), ("point", [str(self.root)]),
            ("descriptor_mount", [18]),
        ):
            original = getattr(self, field)
            for value in values:
                with self.subTest(field=field, value=value):
                    setattr(self, field, value)
                    with self.assertRaises(ValueError):
                        self.measure()
            setattr(self, field, original)
        with self.assertRaises(ValueError):
            driver.read_native_file(str(self.body), self.deadline, (str(self.root)+"/",))
        with self.assertRaises(ValueError):
            driver.read_native_file(str(self.body), self.deadline, (str(self.root)+"/",), binding_sha256="0"*64)

    def test_completed_reads_allow_directory_metadata_drift_without_refreshing_receipt(self):
        binary = self.measure()
        original = copy.deepcopy(binary)
        for changes in (dict(st_size=8192, st_ctime_ns=1),
                        dict(st_nlink=3, st_size=16384, st_mtime_ns=2, st_ctime_ns=3)):
            with self.subTest(changes=changes):
                self.directory_stats.update(changes)
                current = self.measure()
                before = copy.deepcopy(current)
                self.assertNotEqual(current, binary)
                driver.verify_driver_binary(binary, self.deadline, BINDING)
                with patch.object(driver, "read_native_file", return_value=current):
                    driver.verify_driver_binary(binary, self.deadline, BINDING)
                self.assertEqual(binary, original)
                self.assertEqual(current, before)
                self.assertEqual(len(current["driver_mount"]["ancestors"][0][1]), 8)

    def test_directory_metadata_drift_during_one_read_still_refuses(self):
        original_read = os.read
        for field in ("st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"):
            with self.subTest(field=field):
                def drift(fd, count):
                    data = original_read(fd, count)
                    if data:
                        self.directory_stats[field] += 1
                    return data
                with patch.object(driver.os, "read", side_effect=drift):
                    with self.assertRaisesRegex(ValueError, "ancestor changed during read"):
                        self.measure()

    def test_cross_phase_comparison_keeps_stable_ancestors_and_other_evidence_exact(self):
        binary = self.measure()
        original = copy.deepcopy(binary)
        def refuses(changed):
            before = copy.deepcopy(changed)
            with patch.object(driver, "read_native_file", return_value=changed):
                with self.assertRaises(ValueError):
                    driver.verify_driver_binary(binary, self.deadline, BINDING)
            self.assertEqual(changed, before)
            self.assertEqual(binary, original)
        for index in range(4):
            changed = copy.deepcopy(binary)
            changed["driver_mount"]["ancestors"][0][1][index] += 1
            with self.subTest(ancestor_identity=index):
                refuses(changed)
        for kind in ("path", "order", "length"):
            changed = copy.deepcopy(binary)
            rows = changed["driver_mount"]["ancestors"]
            if kind == "path":
                rows[0][0] = "/different"
            elif kind == "order":
                rows.reverse()
            else:
                rows.pop()
            with self.subTest(ancestor_rows=kind):
                refuses(changed)
        for key in binary:
            if key != "driver_mount":
                changed = copy.deepcopy(binary)
                changed[key] = "changed" if isinstance(changed[key], str) else changed[key]+1
                with self.subTest(body=key):
                    refuses(changed)
        for key in binary["driver_mount"]:
            if key != "ancestors":
                changed = copy.deepcopy(binary)
                value = changed["driver_mount"][key]
                changed["driver_mount"][key] = value+["changed"] if isinstance(value, list) else value+"changed"
                with self.subTest(witness=key):
                    refuses(changed)
        for index in range(8):
            changed = copy.deepcopy(binary)
            changed["driver_mount"]["identity"][index] += 1
            with self.subTest(body_identity=index):
                refuses(changed)

    def test_cross_phase_alias_link_identity_and_target_remain_exact(self):
        alias = self.root / "driver-alias"
        alias.symlink_to(self.body)
        with patch.dict(driver.DRIVER_ALIASES, {str(alias): "nvidia_smi"}):
            binary = driver.read_native_file(str(alias), self.deadline,
                (str(self.root)+"/",), binding_sha256=BINDING)
            self.assertTrue(binary["driver_mount"]["links"])
            driver.verify_driver_binary(binary, self.deadline, BINDING)
            for index in range(8):
                changed = copy.deepcopy(binary)
                changed["driver_mount"]["links"][0][1][index] += 1
                with self.subTest(link_identity=index):
                    with patch.object(driver, "read_native_file", return_value=changed):
                        with self.assertRaises(ValueError):
                            driver.verify_driver_binary(binary, self.deadline, BINDING)
            changed = copy.deepcopy(binary)
            changed["driver_mount"]["links"][0][2] += "changed"
            with patch.object(driver, "read_native_file", return_value=changed):
                with self.assertRaises(ValueError):
                    driver.verify_driver_binary(binary, self.deadline, BINDING)

    def test_same_inode_remount_namespace_change_and_stale_declarations_refuse(self):
        binary = self.measure()
        for mutate in (lambda: setattr(self, "mount_id", 18),
                       lambda: self.namespaces.update({"/proc/self/ns/mnt": "mnt:[14]"}),
                       lambda: setattr(self, "mapping", "0 200000 65536\n")):
            mutate()
            with self.assertRaises(ValueError):
                driver.verify_driver_binary(binary, self.deadline, BINDING)
            self.mount_id = self.descriptor_mount = 17
            self.mapping = "0 100000 65536\n"
            self.namespaces["/proc/self/ns/mnt"] = "mnt:[13]"
        with self.assertRaises(ValueError):
            driver.verify_driver_binary(binary, self.deadline, "b"*64)

    def test_during_hash_mount_flip_and_inode_body_mutation_refuse(self):
        original_read = os.read
        def flip(fd, count):
            data = original_read(fd, count)
            if data:
                self.mount_id = 18
            return data
        with patch.object(driver.os, "read", side_effect=flip):
            with self.assertRaises(ValueError):
                self.measure()
        self.mount_id = 17
        binary = self.measure()
        self.body.write_bytes(b"altered driver")
        with self.assertRaises(ValueError):
            driver.verify_driver_binary(binary, self.deadline, BINDING)

    def test_writable_ancestors_hardlinks_alias_escape_and_deadlines_refuse(self):
        self.root.chmod(0o777)
        with self.assertRaises(ValueError):
            self.measure()
        self.root.chmod(0o700)
        hard = self.root / "hardlink"
        os.link(self.body, hard)
        with self.assertRaises(ValueError):
            self.measure()
        hard.unlink()
        alias = self.root / "alias"
        alias.symlink_to(self.body)
        with self.assertRaises(ValueError):
            driver.read_native_file(str(alias), self.deadline, (str(self.root)+"/",), binding_sha256=BINDING)
        self.deadline = time.monotonic()-1
        with self.assertRaises(ValueError):
            self.measure()

    def test_schema_unknown_role_and_namespace_mount_tampering_refuse(self):
        binary = self.measure()
        for key, value in (("role", "interpreter"), ("alias", "/tmp/forged"),
                           ("user_namespace", "user:[bogus]"), ("mount", [1]),
                           ("identity", [1]), ("uid_map", [[0, 0, 65536]]), ("extra", True)):
            changed = copy.deepcopy(binary)
            changed["driver_mount"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.verify_driver_binary(changed, self.deadline, BINDING)

    def test_downstream_uses_protected_session_facts_and_remeasures_qualified_body(self):
        from cineforge.comfy_runtime_worker import verify_runtime_binary
        from cineforge.native_remote_plan import nr_digest, nr_encode
        native = self.root / "native"
        native.mkdir(mode=0o700)
        binding = {"worker": "synthetic-worker", "scope": "a"*64, "session": "synthetic-session"}
        (native / "binding.json").write_bytes(nr_encode(binding))
        (native / "binding.json").chmod(0o600)
        with patch.dict(globals(), BINDING=nr_digest(binding)):
            binary = self.measure()
        (native / "capabilities.json").write_bytes(nr_encode({"binding_sha256": nr_digest(binding),
            "files": [dict(binary, status="measured")]}))
        (native / "capabilities.json").chmod(0o600)
        verify_runtime_binary(binary, native, self.deadline)
        with self.assertRaises(ValueError):
            verify_runtime_binary(binary, None, self.deadline)
        with self.assertRaises((ValueError, NativeSpeechError)):
            verify_runtime_binary({k: binary[k] for k in ("path", "size_bytes", "sha256")}, native, self.deadline)
        (native / "binding.json").write_bytes(nr_encode(dict(binding, worker="foreign-worker")))
        with self.assertRaises(ValueError):
            verify_runtime_binary(binary, native, self.deadline)
        (native / "binding.json").write_bytes(nr_encode(binding))
        self.body.write_bytes(b"changed later binary")
        with self.assertRaises(ValueError):
            verify_runtime_binary(binary, native, self.deadline)

    def test_downstream_original_capture_allows_live_drift_but_not_declaration_tampering(self):
        from cineforge import comfy_runtime_worker as worker
        from cineforge.native_remote_plan import nr_digest, nr_encode
        native = self.root / "native"
        native.mkdir(mode=0o700)
        binding = {"worker": "synthetic-worker", "scope": "a"*64, "session": "synthetic-session"}
        (native / "binding.json").write_bytes(nr_encode(binding))
        (native / "binding.json").chmod(0o600)
        with patch.dict(globals(), BINDING=nr_digest(binding)):
            binary = self.measure()
        captured = nr_encode({"binding_sha256": nr_digest(binding),
            "files": [dict(binary, status="measured")]})
        (native / "capabilities.json").write_bytes(captured)
        (native / "capabilities.json").chmod(0o600)
        original = copy.deepcopy(binary)
        self.directory_stats.update(st_nlink=3, st_size=8192, st_mtime_ns=2, st_ctime_ns=3)
        worker.verify_runtime_binary(binary, native, self.deadline)
        self.assertEqual(binary, original)
        for index in range(4, 8):
            changed = copy.deepcopy(binary)
            changed["driver_mount"]["ancestors"][0][1][index] += 1
            with self.subTest(tampered_captured_field=index):
                with patch.object(worker, "verify_driver_binary", wraps=driver.verify_driver_binary) as verify:
                    with self.assertRaisesRegex(ValueError, "differs from actual consumed worker capabilities"):
                        worker.verify_runtime_binary(changed, native, self.deadline)
                    verify.assert_not_called()
        self.assertEqual((native / "capabilities.json").read_bytes(), captured)

    def test_parent_path_inode_replacement_after_descriptor_open_refuses(self):
        original_read = os.read
        moved = self.root.parent / (self.root.name+"-moved")
        triggered = False
        def replace_parent(fd, count):
            nonlocal triggered
            data = original_read(fd, count)
            if data and not triggered:
                triggered = True
                self.root.rename(moved)
                self.root.mkdir(mode=0o700)
                (self.root / self.body.name).write_bytes(data)
            return data
        try:
            with patch.object(driver.os, "read", side_effect=replace_parent):
                with self.assertRaises((ValueError, OSError)):
                    self.measure()
        finally:
            if triggered:
                import shutil
                shutil.rmtree(self.root)
                moved.rename(self.root)

    def test_capability_controller_and_reviewed_plan_reject_mixed_or_stale_witness(self):
        from test_comfy_runtime import Reviewer, plan

        from cineforge import native_remote as controller
        from cineforge.comfy_runtime import CheckedComfyRuntime, ClosedComfyRuntime
        from cineforge.comfy_runtime_plan import comfy_plan
        from cineforge.native_remote import RemoteNativeSpeechRuntime
        from cineforge.native_remote_plan import nr_digest, nr_encode
        binding = {"worker": "owned-worker", "session": "owned-session", "scope": "c"*64}
        with patch.dict(globals(), BINDING=nr_digest(binding)):
            binary = self.measure()
        bridge = object.__new__(RemoteNativeSpeechRuntime)
        bridge._binding = binding
        facts = {"version": 1, "policy": controller.NR_CAP_POLICY, "binding_sha256": nr_digest(binding),
                 "interpreter_path": "/usr/bin/python3.11", "files": [dict(binary, role="nvidia_smi", alias=str(self.body), status="measured"),
                 dict(role="interpreter", alias="/usr/bin/python3.11", status="measured", path="/usr/bin/python3.11", size_bytes=1, sha256="d"*64)]}
        aliases = (("nvidia_smi", str(self.body)), ("interpreter", "/usr/bin/python3.11"))
        with patch.object(controller, "NR_CAP_ALIASES", aliases), patch.object(controller, "NR_CAP_ROOTS", (str(self.root)+"/", "/usr/bin/")):
            bridge._validate_capabilities(facts)
            tampered = copy.deepcopy(facts)
            tampered["files"][0]["driver_mount"]["role"] = "interpreter"
            with self.assertRaises(ValueError):
                bridge._validate_capabilities(tampered)
        declaration = plan()
        declaration["native"].append(binary)
        declaration["native"].sort(key=lambda b: b["path"])
        comfy_plan(declaration)
        worker = SimpleNamespace(spec=SimpleNamespace(image=declaration["image"]),
            selection=SimpleNamespace(worker_profile={"id": "m5-wan22-s2v14b-5090-v1"}), native_admission=lambda: None)
        bridge._worker = worker
        bridge._deadline = self.deadline
        bridge._expires_unix = time.time()+10
        bridge._installed = {"interpreter": declaration["interpreter"]}
        bridge._capabilities_payload = nr_encode(facts)
        bridge._capabilities_sha256 = nr_digest(facts)
        checked = CheckedComfyRuntime.reviewed(declaration, Reviewer())
        closed = ClosedComfyRuntime(bridge, checked)
        closed._admit(worker, self.deadline)
        original_plan = copy.deepcopy(checked.current())
        for index in range(4, 8):
            changed = copy.deepcopy(declaration)
            row = next(row for row in changed["native"] if "driver_mount" in row)
            row["driver_mount"]["ancestors"][0][1][index] += 1
            mutated = ClosedComfyRuntime(bridge, CheckedComfyRuntime.reviewed(changed, Reviewer()))
            with self.subTest(tampered_checked_field=index), self.assertRaises(ValueError):
                mutated._admit(worker, self.deadline)
        self.assertEqual(checked.current(), original_plan)
        bridge._capabilities_payload = None
        with self.assertRaises(ValueError):
            closed._admit(worker, self.deadline)
        bridge._capabilities_payload = nr_encode(facts)
        bridge._binding = dict(binding, session="foreign-session")
        with self.assertRaises(ValueError):
            closed._admit(worker, self.deadline)
