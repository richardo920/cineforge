"""Synthetic ZIP/tar/owned-process fixtures; no actual wheel, SDK, GPU or readiness proof."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import os
import stat
import tarfile
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cineforge import comfy_runtime_launcher as launcher
from cineforge import comfy_runtime_worker as worker
from cineforge.comfy_runtime import CheckedComfyRuntime, ClosedComfyRuntime, ComfyClosureReview
from cineforge.comfy_runtime_plan import (
    COMFY_ABI,
    COMFY_EXECUTION,
    COMFY_FIXED,
    COMFY_POLICY,
    COMFY_REQUIREMENTS_SHA256,
    COMFY_ROOTS,
    COMFY_SOURCE,
    comfy_plan,
    comfy_subject,
)
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_remote_plan import nr_digest, nr_encode
from cineforge.native_remote_worker import _install_wheel
from cineforge.native_speech_integrity import NativeBinary, NativeSpeechError, verify_binary
from cineforge.s2v_profile import IMAGE, model_files
from cineforge.worker_bundle import worker_source

HASH = "a" * 64
CANARY = "SYNTHETIC-PRIVATE-CANARY"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wheel(
    name="fixture", version="1", *, additional=None, changes=None, record_changes=None, special=None
):
    directory = name.replace("-", "_") + "-" + version + ".dist-info/"
    bodies = {
        name.replace("-", "_")
        + "/__init__.py": b"raise AssertionError('synthetic code must not import')\n",
        directory
        + "METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
        directory + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        directory + "NOTICE": b"Synthetic fixture notice only.\n",
    }
    if name == "comfyui-frontend-package":
        bodies["comfyui_frontend_package/static/index.html"] = b"synthetic frontend"
    bodies.update(additional or {})
    bodies.update(changes or {})
    rows = [
        [
            path,
            "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("="),
            str(len(raw)),
        ]
        for path, raw in sorted(bodies.items())
    ]
    if record_changes:
        record_changes(rows)
    rows.append([directory + "RECORD", "", ""])
    text = io.StringIO(newline="")
    csv.writer(text).writerows(rows)
    bodies[directory + "RECORD"] = text.getvalue().encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for path, raw in bodies.items():
            if special and path == special[0]:
                info = zipfile.ZipInfo(path)
                info.create_system = 3
                info.external_attr = special[1] << 16
                archive.writestr(info, raw)
            else:
                archive.writestr(path, raw)
    raw = output.getvalue()
    filename = name.replace("-", "_") + "-" + version + "-py3-none-any.whl"
    declaration = dict(
        name=name,
        version=version,
        filename=filename,
        url="https://files.pythonhosted.org/packages/a/" + filename,
        size_bytes=len(raw),
        sha256=sha(raw),
        metadata_sha256=sha(bodies[directory + "METADATA"]),
        wheel_sha256=sha(bodies[directory + "WHEEL"]),
        requires_python="",
        requires_dist=[],
        tags=["py3-none-any"],
        notices=[
            [
                directory + "NOTICE",
                len(bodies[directory + "NOTICE"]),
                sha(bodies[directory + "NOTICE"]),
            ]
        ],
    )
    return raw, declaration, bodies


def plan():
    wheels, manifest = [], {}
    for name in sorted(COMFY_ROOTS | set(COMFY_FIXED)):
        _, declaration, bodies = wheel(name, COMFY_FIXED.get(name, "1"))
        wheels.append(declaration)
        manifest.update({"site/" + name: body for name, body in bodies.items()})
    files = [[name, 1, HASH] for name in ("LICENSE", "comfy/cli_args.py", "main.py", "nodes.py")]
    files.append(["requirements.txt", 529, COMFY_REQUIREMENTS_SHA256])
    return dict(
        version=1,
        policy=COMFY_POLICY,
        image=IMAGE,
        abi=COMFY_ABI,
        source=dict(revision=COMFY_SOURCE, size_bytes=1, sha256=HASH, files=sorted(files)),
        wheels=wheels,
        installed=[[name, len(raw), sha(raw)] for name, raw in sorted(manifest.items())],
        interpreter=dict(path="/closed-fixture/python", size_bytes=1, sha256=HASH),
        native=[dict(path="/usr/bin/nvidia-smi", size_bytes=1, sha256=HASH)],
        review={
            key: HASH for key in ("artifacts", "dependencies", "licenses", "advisories", "abi")
        },
        notices=[],
        execution=dict(COMFY_EXECUTION),
        shared_files=[],
        executables=[],
    )


class Reviewer:
    """Synthetic trusted callback; never actual artifact, license or ABI clearance."""

    def __init__(self):
        self.calls = 0
        self.accept = True

    def verify(self, declaration):
        self.calls += 1
        if not self.accept:
            raise RuntimeError(CANARY)
        return ComfyClosureReview(
            comfy_subject(declaration),
            *(
                declaration["review"][key]
                for key in ("artifacts", "dependencies", "licenses", "advisories", "abi")
            ),
        )


class ComfyPlanTests(unittest.TestCase):
    def test_explicit_actual_review_is_required_fresh_and_bound_to_all_plan_bytes(self):
        declaration = plan()
        reviewer = Reviewer()
        checked = CheckedComfyRuntime.reviewed(declaration, reviewer)
        self.assertEqual(reviewer.calls, 1)
        declaration["wheels"][0]["sha256"] = "b" * 64
        self.assertNotEqual(checked.current(), declaration)
        self.assertEqual(reviewer.calls, 2)
        reviewer.accept = False
        with self.assertRaises(ValueError) as caught:
            checked.current()
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn(CANARY, str(caught.exception))
        for verifier in (None, {"approved": True}, SimpleNamespace(verify=lambda _: True)):
            with self.assertRaises(ValueError):
                CheckedComfyRuntime.reviewed(plan(), verifier)
        forged = object.__new__(CheckedComfyRuntime)
        with self.assertRaises(ValueError):
            forged.current()

    def test_wrong_missing_duplicate_version_tag_origin_metadata_and_identity_refuse(self):
        mutations = [
            lambda p: p.update(version=True),
            lambda p: p.update(image="unreviewed"),
            lambda p: p.update(abi="python3"),
            lambda p: p["source"].update(revision="main"),
            lambda p: p["source"]["files"].pop(),
            lambda p: p["wheels"].pop(),
            lambda p: p["wheels"].append(p["wheels"][0]),
            lambda p: p["wheels"][0].update(url="https://unreviewed.example/runtime.whl"),
            lambda p: p["wheels"][0].update(size_bytes=True),
            lambda p: p["wheels"][0].update(tags=["cp314-cp314-any"]),
            lambda p: p["wheels"][0].update(metadata_sha256="unknown"),
            lambda p: p["wheels"][0].update(notices=[]),
            lambda p: p["wheels"][0].update(requires_dist=["z", "a"]),
            lambda p: p.update(installed=[]),
            lambda p: p.update(native=[]),
            lambda p: p["interpreter"].update(size_bytes=True),
            lambda p: p["review"].update(abi="unknown"),
            lambda p: p["wheels"][
                next(i for i, w in enumerate(p["wheels"]) if w["name"] == "torch")
            ].update(version="2.9.0"),
        ]
        for change in mutations:
            with self.subTest(change=change):
                declaration = plan()
                change(declaration)
                reviewer = Reviewer()
                with self.assertRaises((ValueError, TypeError)):
                    CheckedComfyRuntime.reviewed(declaration, reviewer)
                self.assertEqual(reviewer.calls, 0)

    def test_default_import_and_declaration_check_have_no_io_or_native_import(self):
        declaration = plan()
        with (
            patch("builtins.open", side_effect=AssertionError("file")),
            patch("os.open", side_effect=AssertionError("file")),
            patch("socket.socket", side_effect=AssertionError("network")),
            patch("subprocess.Popen", side_effect=AssertionError("child")),
        ):
            self.assertEqual(comfy_plan(declaration), declaration)
            CheckedComfyRuntime.reviewed(declaration, Reviewer()).current()
        import sys

        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("av", sys.modules)


class ComfyExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cf-comfy-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.target = self.root / "runtime"
        self.target.mkdir(mode=0o700)
        self.deadline = time.monotonic() + 5

    def install(self, raw, declaration, manifest=None):
        path = self.root / (declaration["filename"] + str(len(list(self.root.iterdir()))))
        path.write_bytes(raw)
        path.chmod(0o600)
        result = {} if manifest is None else manifest
        worker.install_wheel(path, self.target, declaration, result, self.deadline)
        return result

    def test_whole_wheel_record_headers_notices_and_exclusive_manifest_are_verified(self):
        raw, declaration, bodies = wheel()
        manifest = self.install(raw, declaration)
        self.assertEqual(
            manifest, {"site/" + name: (len(body), sha(body)) for name, body in bodies.items()}
        )
        with self.assertRaises(ValueError):
            self.install(raw, declaration, manifest)
        self.assertTrue(
            all((self.target / "site" / name).read_bytes() == body for name, body in bodies.items())
        )

    def test_nested_vendor_distribution_and_pth_are_retained_as_data_without_execution(self):
        raw, declaration, bodies = wheel(
            "setuptools",
            "84.2.1",
            additional={
                "setuptools/_vendor/example-1.dist-info/METADATA": b"Name: example\nVersion: 1\n",
                "setuptools/_vendor/example-1.dist-info/RECORD": b"inert nested fixture",
                "distutils-precedence.pth": b"import builtins;raise AssertionError('must not execute')\n",
            },
        )
        result = self.install(raw, declaration)
        self.assertIn("site/distutils-precedence.pth", result)
        self.assertEqual(
            (self.target / "site/distutils-precedence.pth").read_bytes(),
            bodies["distutils-precedence.pth"],
        )
        # Native108's default remains strict, not relaxed to accept Comfy's extras.
        path = self.root / "legacy.whl"
        path.write_bytes(raw)
        path.chmod(0o600)
        with self.assertRaises(ValueError):
            _install_wheel(path, self.target, declaration, {}, self.deadline)

    def test_wrong_bytes_metadata_tags_notices_and_record_refuse_before_imports(self):
        raw, declaration, _ = wheel()
        for field, wrong in (
            ("sha256", "b" * 64),
            ("metadata_sha256", "b" * 64),
            ("wheel_sha256", "b" * 64),
            ("tags", ["cp311-cp311-any"]),
            ("version", "2"),
            ("requires_dist", ["foreign"]),
            ("notices", [["fixture-1.dist-info/NOTICE", 1, HASH]]),
        ):
            with self.subTest(field=field):
                with self.assertRaises((ValueError, KeyError)):
                    self.install(raw, dict(declaration, **{field: wrong}))
        bad, declaration, _ = wheel(
            record_changes=lambda rows: rows[0].__setitem__(1, "sha256=" + "A" * 43)
        )
        with self.assertRaises(ValueError):
            self.install(bad, declaration)

    def test_paths_links_overwrites_duplicate_destinations_and_expiry_refuse(self):
        for additional, special in (
            ({"../escape": b"x"}, None),
            ({"/escape": b"x"}, None),
            ({"fixture.data/purelib/fixture/__init__.py": b"same"}, None),
            ({"link": b"destination"}, ("link", stat.S_IFLNK | 0o777)),
        ):
            raw, declaration, _ = wheel(additional=additional, special=special)
            with self.assertRaises((ValueError, FileExistsError)):
                self.install(raw, declaration)
        self.assertFalse((self.root / "escape").exists())
        self.deadline = time.monotonic() - 1
        raw, declaration, _ = wheel()
        with self.assertRaises(ValueError):
            self.install(raw, declaration)

    def test_native_binary_hash_link_and_full_installed_target_mutations_refuse(self):
        binary = self.root / "binary"
        binary.write_bytes(b"synthetic-native-body")
        binary.chmod(0o600)
        pin = NativeBinary(str(binary), binary.stat().st_size, sha(binary.read_bytes()))
        verify_binary(pin, self.deadline)
        binary.write_bytes(b"tampered-native-body!")
        with self.assertRaises(ValueError):
            verify_binary(pin, self.deadline)
        binary.unlink()
        binary.symlink_to(self.root / "foreign")
        with self.assertRaises((ValueError, OSError)):
            verify_binary(pin, self.deadline)

    def test_source_archive_hash_full_files_traversal_links_and_duplicate_refuse(self):
        def archive(names):
            output = io.BytesIO()
            with tarfile.open(fileobj=output, mode="w:gz") as writer:
                for name, body, kind in names:
                    member = tarfile.TarInfo("ComfyUI-" + COMFY_SOURCE + "/" + name)
                    member.type = kind
                    member.size = len(body)
                    writer.addfile(member, io.BytesIO(body))
            raw = output.getvalue()
            return raw, dict(
                size_bytes=len(raw), sha256=sha(raw), files=[["main.py", 1, sha(b"x")]]
            )

        for index, names in enumerate(
            (
                [("main.py", b"x", tarfile.REGTYPE)],
                [("main.py", b"x", tarfile.REGTYPE), ("../escape", b"x", tarfile.REGTYPE)],
                [("main.py", b"x", tarfile.SYMTYPE)],
                [("main.py", b"x", tarfile.REGTYPE), ("main.py", b"x", tarfile.REGTYPE)],
            )
        ):
            raw, declaration = archive(names)
            path = self.root / f"source{index}.tar.gz"
            path.write_bytes(raw)
            path.chmod(0o600)
            target = self.root / f"source{index}"
            target.mkdir(mode=0o700)
            if index == 0:
                worker.install_source(path, target, declaration, self.deadline)
                self.assertEqual((target / "main.py").read_bytes(), b"x")
                with self.assertRaises(ValueError):
                    worker.install_source(
                        path, target, dict(declaration, sha256="b" * 64), self.deadline
                    )
            else:
                with self.assertRaises(ValueError):
                    worker.install_source(path, target, declaration, self.deadline)
        self.assertFalse((self.root / "escape").exists())


class ComfyControllerTests(unittest.TestCase):
    def fixture(self):
        declaration = plan()
        reviewer = Reviewer()
        checked = CheckedComfyRuntime.reviewed(declaration, reviewer)
        state = {"revoked": False, "calls": [], "lost": None}

        def gate(*args):
            if state["revoked"]:
                raise ValueError(CANARY)

        worker = SimpleNamespace(
            spec=SimpleNamespace(image=IMAGE), native_admission=gate, _require_s2v=gate
        )
        bridge = object.__new__(RemoteNativeSpeechRuntime)
        bridge._worker = worker
        bridge._deadline = time.monotonic() + 100
        bridge._expires_unix = time.time() + 100
        bridge._binding = {}
        bridge._installed = {"interpreter": declaration["interpreter"]}
        bridge._unresolved = False
        selected = SimpleNamespace(declarations={"model": {"files": model_files()}})

        def upload(*args, **kwargs):
            state["calls"].append("transfer")
            if state["lost"] == "transfer":
                raise TimeoutError(CANARY)

        def call(operation, **kwargs):
            state["calls"].append(operation)
            if state["lost"] == operation:
                raise TimeoutError(CANARY)
            if operation == "native_comfy_model":
                item = selected.declarations["model"]["files"][kwargs["model_index"]]
                return {key: item[key] for key in ("path", "size_bytes", "sha256")}
            if operation == "native_comfy_setup":
                return dict(
                    verification={},
                    runtime={},
                    comfy={
                        "subject_sha256": comfy_subject(declaration),
                        "installed_sha256": nr_digest(declaration["installed"]),
                        "probe_sha256": HASH,
                        "execution_sha256": nr_digest(declaration["execution"]),
                        "jit_enabled": False,
                        "phase": "after_core_startup",
                    },
                )
            return {}

        bridge._upload = upload
        bridge._call = call
        runtime = ClosedComfyRuntime(bridge, checked)
        return runtime, worker, selected, reviewer, state, bridge

    def test_closed_once_phases_preserve_current_rights_and_original_cap(self):
        runtime, worker, selected, reviewer, state, bridge = self.fixture()
        cap = time.monotonic() + 30
        runtime.prepare(worker, selected, deadline=cap)
        self.assertEqual(bridge._deadline, cap)
        self.assertEqual(
            state["calls"],
            [
                "transfer",
                "native_comfy_install",
                *(["native_comfy_model"] * 4),
                "native_comfy_setup",
            ],
        )
        self.assertGreater(reviewer.calls, 7)
        with self.assertRaises(ValueError):
            runtime.prepare(worker, selected, deadline=cap + 100)
        self.assertEqual(bridge._deadline, cap)

    def test_revocation_after_install_blocks_first_model_and_lost_ack_never_replays(self):
        for loss in (
            "transfer",
            "native_comfy_install",
            "native_comfy_model",
            "native_comfy_setup",
        ):
            runtime, worker, selected, reviewer, state, bridge = self.fixture()
            state["lost"] = loss
            with self.assertRaises(ValueError):
                runtime.prepare(worker, selected, deadline=time.monotonic() + 30)
            calls = list(state["calls"])
            with self.assertRaises(ValueError):
                runtime.prepare(worker, selected, deadline=time.monotonic() + 100)
            self.assertEqual(calls, state["calls"])
            self.assertTrue(bridge._unresolved)
        runtime, worker, selected, reviewer, state, bridge = self.fixture()
        original = bridge._call

        def revoke(op, **kwargs):
            result = original(op, **kwargs)
            if op == "native_comfy_install":
                state["revoked"] = True
            return result

        bridge._call = revoke
        with self.assertRaises(ValueError):
            runtime.prepare(worker, selected, deadline=time.monotonic() + 30)
        self.assertNotIn("native_comfy_model", state["calls"])
        self.assertTrue(bridge._unresolved)

    def test_missing_stale_or_changed_worker_identity_refuses_without_transfer(self):
        for kind in ("review", "image", "interpreter", "expiry"):
            runtime, worker, selected, reviewer, state, bridge = self.fixture()
            cap = time.monotonic() + 30
            if kind == "review":
                reviewer.accept = False
            if kind == "image":
                worker.spec.image = "unknown"
            if kind == "interpreter":
                bridge._installed = {"interpreter": {}}
            if kind == "expiry":
                cap = time.monotonic() - 1
            with self.assertRaises(ValueError):
                runtime.prepare(worker, selected, deadline=cap)
            self.assertEqual(state["calls"], [])

    def test_launcher_has_no_site_no_ambient_environment_and_first_party_source_is_bound(self):
        declaration = plan()
        command, environment = launcher.launch_command(
            declaration, Path("/fixture"), Path("/native"), "probe", time.monotonic() + 5
        )
        self.assertEqual(command[1:5], ["-I", "-B", "-S", "-c"])
        for key in ("PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "HTTP_PROXY", "HF_TOKEN"):
            self.assertNotIn(key, environment)
        self.assertEqual(environment["PATH"], "")
        source = worker_source()
        self.assertLess(len(source), 128 * 1024)
        self.assertIn(b'"comfy"', source)
        self.assertIn(b"hardware_only_unverified_runtime", source)


class ComfyBoundaryTests(unittest.TestCase):
    def test_full_installed_manifest_rejects_extra_tamper_symlink_hardlink_and_nested_counts(self):
        with tempfile.TemporaryDirectory(prefix="cf-comfy-installed-") as temporary:
            root = Path(temporary).resolve()
            runtime = root / "comfy-runtime"
            runtime.mkdir(mode=0o700)
            (root / "comfy-notices").mkdir(mode=0o700)
            (root / "comfy-notices-manifest.json").write_bytes(nr_encode([]))
            (root / "comfy-notices-manifest.json").chmod(0o600)
            declaration = plan()
            manifest = {}
            for selected in declaration["wheels"]:
                raw, _, _ = wheel(selected["name"], selected["version"])
                path = root / selected["filename"]
                path.write_bytes(raw)
                path.chmod(0o600)
                worker.install_wheel(path, runtime, selected, manifest, time.monotonic() + 5)
            with patch.object(worker, "verify_binary"):
                worker.verify_runtime(declaration, root, time.monotonic() + 5)
                relative = declaration["installed"][0][0]
                target = runtime / relative
                original = target.read_bytes()
                for fault in ("tamper", "extra", "symlink", "hardlink"):
                    with self.subTest(fault=fault):
                        if fault == "tamper":
                            target.write_bytes(b"x" * len(original))
                        if fault == "extra":
                            (runtime / "unknown.py").write_bytes(b"x")
                        if fault == "symlink":
                            target.unlink()
                            target.symlink_to(root / "foreign")
                        if fault == "hardlink":
                            os.link(target, root / "alias")
                        with self.assertRaises((ValueError, OSError)):
                            worker.verify_runtime(declaration, root, time.monotonic() + 5)
                        (runtime / "unknown.py").unlink(missing_ok=True)
                        (root / "alias").unlink(missing_ok=True)
                        target.unlink(missing_ok=True)
                        target.write_bytes(original)
                        target.chmod(0o600)

    def test_owned_no_site_startup_never_processes_executable_pth_or_ambient_hooks(self):
        import subprocess
        import sys

        from cineforge.native_remote_package import native_package
        from cineforge.native_remote_plan import nr_json

        with tempfile.TemporaryDirectory(prefix="cf-comfy-nosite-") as temporary:
            root = Path(temporary).resolve()
            native = root / "native"
            package_root = native / "firstparty/cineforge"
            package_root.mkdir(parents=True, mode=0o700)
            for name, body in nr_json(native_package()[0]).items():
                (package_root / name).write_bytes(base64.b64decode(body))
            marker = root / "executed-unreviewed-startup"
            code = f"import pathlib;pathlib.Path({str(marker)!r}).write_text('executed')\n"
            ambient = root / "ambient"
            ambient.mkdir()
            (ambient / "sitecustomize.py").write_text(code)
            (ambient / "executable.pth").write_text(code)
            runtime = root / "comfy-runtime"
            runtime.mkdir(mode=0o700)
            (runtime / "site").mkdir(mode=0o700)
            # Executable startup code is retained as a file, never benign or dropped.
            (runtime / "site/executable.pth").write_text(code)
            (runtime / "site/torch.py").write_text(code)
            declaration = plan()
            executable = Path(sys.executable).resolve()
            declaration["interpreter"] = dict(
                path=str(executable),
                size_bytes=executable.stat().st_size,
                sha256=sha(executable.read_bytes()),
            )
            (root / "comfy-selection.json").write_bytes(nr_encode(declaration))
            (root / "comfy-selection.json").chmod(0o600)
            command, environment = launcher.launch_command(
                declaration, root, native, "probe", time.monotonic() + 5
            )
            environment.update(PYTHONPATH=str(ambient), PYTHONUSERBASE=str(ambient))
            result = subprocess.run(
                command, env=environment, input=b"", capture_output=True, timeout=8
            )
            self.assertEqual(
                result.returncode, 2
            )  # Invalid staged native identities refuse before Torch.
            self.assertEqual(result.stdout, b"")
            self.assertEqual(result.stderr, b"")
            self.assertFalse(marker.exists())

    def test_m5_arm_evidence_is_nvml_only_and_cannot_be_measured_torch_facts(self):
        import platform
        import sys

        from test_conditioning import runtime_fixture

        from cineforge import worker_bootstrap as bootstrap
        from cineforge.ssh_worker import SSHWorker, WorkerError

        runtime = runtime_fixture()
        resources = {key: runtime[key] for key in bootstrap.M4_RESOURCE_FIELDS}
        raw = b"NVIDIA GeForce RTX 5090,32768,64,12.0,580.95.05\n"
        with (
            patch.object(bootstrap, "bounded_process", return_value=raw) as child,
            patch.object(bootstrap, "resource_evidence", return_value=resources),
            patch.object(sys, "version_info", (3, 11, 14)),
            patch.object(sys, "platform", "linux"),
            patch.object(platform, "machine", return_value="x86_64"),
            patch.object(platform, "python_version", return_value="3.11.14"),
        ):
            result = bootstrap.s2v_hardware_admission(Path("/fixture"), time.monotonic() + 5)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(child.call_args.args[0][0], "/usr/bin/nvidia-smi")
        self.assertNotIn("torch_version", result)
        self.assertNotIn("compute_precision", result)
        self.assertEqual(result["admission_status"], "hardware_only_unverified_runtime")
        self.assertEqual(SSHWorker._runtime_evidence(result, s2v=True, admission=True), result)
        with self.assertRaises(WorkerError):
            SSHWorker._runtime_evidence(
                dict(runtime, compute_precision="bf16"), s2v=True, admission=True
            )
        with self.assertRaises(ValueError):
            bootstrap.validate_s2v_runtime(result)

    def test_comfy_transfer_reuses_fixed_integrity_protocol_and_original_local_cap(self):
        import test_native_remote as native_fixtures

        from cineforge import native_remote_bootstrap as bootstrap

        fixture = native_fixtures.RemoteFixture("runTest")
        fixture.setUp()
        try:
            fixture.prepare()
            raw = nr_encode(plan())
            cutoff = min(fixture.bridge._expires_unix, time.time() + 30)
            fixture.bridge._upload("comfy", raw, comfy_expires_unix=cutoff)
            self.assertEqual(bootstrap.nr_read(fixture.remote / "comfy.blob"), raw)
            local = bootstrap.nr_json(bootstrap.nr_read(fixture.remote / "comfy-local-cap.json"))
            self.assertEqual(local["expires_unix"], cutoff)
            self.assertLessEqual(local["deadline"], time.monotonic() + 25)
            self.assertTrue((fixture.remote / "comfy.blob.done").exists())
            with self.assertRaises(NativeSpeechError):
                fixture.bridge._upload("comfy", raw, comfy_expires_unix=cutoff + 1)
            self.assertTrue(fixture.bridge._unresolved)
            self.assertEqual(fixture.child_count, 0)
        finally:
            fixture.doCleanups()

    def test_comfy_transfer_lost_chunk_ack_and_oversize_begin_are_consumed(self):
        import test_native_remote as native_fixtures

        from cineforge.native_remote_plan import NR_BLOB_MAX

        for fault in ("lost", "oversize", "bad-cap"):
            fixture = native_fixtures.RemoteFixture("runTest")
            fixture.setUp()
            try:
                fixture.prepare()
                cutoff = min(fixture.bridge._expires_unix, time.time() + 30)
                if fault == "lost":
                    fixture.lost = "native_chunk"

                    def action():
                        return fixture.bridge._upload(
                            "comfy", nr_encode(plan()), comfy_expires_unix=cutoff
                        )
                else:

                    def action():
                        return fixture.bridge._call(
                            "native_begin",
                            role="comfy",
                            token="b" * 32,
                            size_bytes=NR_BLOB_MAX + 1 if fault == "oversize" else 1,
                            sha256=HASH,
                            comfy_expires_unix=True if fault == "bad-cap" else cutoff,
                        )

                with self.assertRaises(NativeSpeechError):
                    action()
                before = len(fixture.requests)
                with self.assertRaises(NativeSpeechError):
                    fixture.bridge._upload("comfy", nr_encode(plan()), comfy_expires_unix=cutoff)
                self.assertEqual(before, len(fixture.requests))
                self.assertTrue(fixture.bridge._unresolved)
                self.assertEqual(fixture.child_count, 0)
                if fault == "lost":
                    self.assertFalse((fixture.remote / "comfy.blob.done").exists())
                else:
                    self.assertTrue((fixture.remote / "failed").exists())
            finally:
                fixture.doCleanups()


class ComfyNoticeAndExecutionTests(unittest.TestCase):
    def test_external_notice_bytes_are_bound_without_url_fetch_or_import_namespace(self):
        declaration = plan()
        selected = declaration["wheels"][0]
        selected["notices"] = []
        body = b"Synthetic notice, no distribution or ABI clearance.\n"
        notice = {
            "source_uri": "https://example.test/repository/immutable/LICENSE",
            "source_revision": "a" * 40,
            "packages": [{"name": selected["name"], "version": selected["version"]}],
            "size_bytes": len(body),
            "sha256": sha(body),
            "content_base64": base64.b64encode(body).decode(),
        }
        declaration["notices"] = [notice]
        with (
            patch("urllib.request.build_opener", side_effect=AssertionError("network")),
            patch("subprocess.Popen", side_effect=AssertionError("child")),
        ):
            checked = CheckedComfyRuntime.reviewed(declaration, Reviewer())
            self.assertEqual(checked.current()["notices"], [notice])
        original = comfy_subject(declaration)
        for field, bad in (
            ("sha256", "b" * 64),
            ("size_bytes", True),
            ("source_uri", "https://user:secret@example.test/LICENSE"),
            ("source_revision", "PRIVATE-CANARY/path"),
            ("content_base64", "INVALID"),
            ("packages", [{"name": selected["name"], "version": "unreviewed"}]),
        ):
            changed = copy.deepcopy(declaration)
            changed["notices"][0][field] = bad
            reviewer = Reviewer()
            with self.assertRaises(ValueError) as caught:
                CheckedComfyRuntime.reviewed(changed, reviewer)
            self.assertEqual(reviewer.calls, 0)
            self.assertIsNone(caught.exception.__context__)
            self.assertNotIn("secret", str(caught.exception))
        changed = copy.deepcopy(declaration)
        changed["notices"][0]["source_revision"] = "b" * 40
        self.assertNotEqual(comfy_subject(changed), original)

    def test_jit_off_is_fixed_preimport_policy_and_actual_state_must_be_false(self):
        declaration = plan()
        command, environment = launcher.launch_command(
            declaration, Path("/fixture"), Path("/native"), "probe", time.monotonic() + 5
        )
        self.assertEqual(environment["PYTORCH_JIT"], "0")
        self.assertIn("-S", command)
        torch = SimpleNamespace(jit=SimpleNamespace(_state=SimpleNamespace(_enabled=False)))
        with patch.dict(os.environ, {"PYTORCH_JIT": "0"}):
            launcher._require_jit_disabled(torch)
            torch.jit._state._enabled = True
            with self.assertRaises(ValueError):
                launcher._require_jit_disabled(torch)
        torch.jit._state._enabled = False
        with patch.dict(os.environ, {"PYTORCH_JIT": "1"}):
            with self.assertRaises(ValueError):
                launcher._require_jit_disabled(torch)
        changed = copy.deepcopy(declaration)
        changed["execution"]["PYTORCH_JIT"] = "1"
        reviewer = Reviewer()
        with self.assertRaises(ValueError):
            CheckedComfyRuntime.reviewed(changed, reviewer)
        self.assertEqual(reviewer.calls, 0)
        value = {
            "runtime": {},
            "imports": {},
            "codecs": {},
            "abi": COMFY_ABI,
            "native_files": {},
            "execution_sha256": nr_digest(COMFY_EXECUTION),
            "jit_enabled_before": False,
            "jit_enabled_after": True,
        }
        with patch("cineforge.worker_bootstrap.resource_evidence") as resources:
            with self.assertRaises(ValueError):
                launcher.probe_result(value, Path("/fixture"), declaration, time.monotonic() + 5)
            resources.assert_not_called()

    def test_only_explicit_empty_nvidia_namespace_can_be_shared_without_overwrite(self):
        from cineforge.comfy_runtime_plan import COMFY_SHARED_EMPTY_SHA256, COMFY_SHARED_OWNERS

        shared = {
            "path": "nvidia/__init__.py",
            "size_bytes": 0,
            "sha256": COMFY_SHARED_EMPTY_SHA256,
            "owners": list(COMFY_SHARED_OWNERS),
        }
        with tempfile.TemporaryDirectory(prefix="cf-comfy-shared-") as temporary:
            root = Path(temporary).resolve()
            target = root / "runtime"
            target.mkdir(mode=0o700)
            manifest = {}
            inode = None
            for name in COMFY_SHARED_OWNERS[:2]:
                raw, declaration, _ = wheel(name, additional={"nvidia/__init__.py": b""})
                path = root / declaration["filename"]
                path.write_bytes(raw)
                path.chmod(0o600)
                worker.install_wheel(
                    path,
                    target,
                    declaration,
                    manifest,
                    time.monotonic() + 5,
                    shared_files=(shared,),
                )
                current = (target / "site/nvidia/__init__.py").stat().st_ino
                if inode is not None:
                    self.assertEqual(
                        current, inode
                    )  # Existing owned file was verified, not overwritten.
                inode = current
            (target / "site/nvidia/__init__.py").write_bytes(b"tamper")
            name = COMFY_SHARED_OWNERS[2]
            raw, declaration, _ = wheel(name, additional={"nvidia/__init__.py": b""})
            path = root / declaration["filename"]
            path.write_bytes(raw)
            path.chmod(0o600)
            with self.assertRaises(ValueError):
                worker.install_wheel(
                    path,
                    target,
                    declaration,
                    manifest,
                    time.monotonic() + 5,
                    shared_files=(shared,),
                )
        declaration = plan()
        declaration["shared_files"] = [shared]
        with self.assertRaises(ValueError):  # Missing exact owners cannot declare a shared target.
            CheckedComfyRuntime.reviewed(declaration, Reviewer())


class ComfyPreimportTests(unittest.TestCase):
    def test_unsupported_actual_abi_refuses_before_any_third_party_import(self):
        import importlib
        import sys

        with (
            patch.object(sys, "version_info", (3, 14, 0)),
            patch.object(importlib, "import_module") as imports,
        ):
            with self.assertRaises(ValueError):
                launcher._observe(Path("/fixture"), time.monotonic() + 5)
            imports.assert_not_called()

    def test_exact_manpage_is_retained_outside_site_and_other_data_schemes_refuse(self):
        with tempfile.TemporaryDirectory(prefix="cf-comfy-data-") as temporary:
            root = Path(temporary).resolve()
            target = root / "runtime"
            target.mkdir(mode=0o700)
            member = "sympy-1.14.0.data/data/share/man/man1/isympy.1"
            raw, declaration, bodies = wheel(
                "sympy", "1.14.0", additional={member: b"synthetic manpage"}
            )
            path = root / declaration["filename"]
            path.write_bytes(raw)
            path.chmod(0o600)
            manifest = {}
            worker.install_wheel(path, target, declaration, manifest, time.monotonic() + 5)
            data = target / "data/sympy-1.14.0/isympy.1"
            self.assertEqual(data.read_bytes(), bodies[member])
            self.assertIn("data/sympy-1.14.0/isympy.1", manifest)
            self.assertFalse((target / "site" / member).exists())
            for member in ("sympy-1.14.0.data/scripts/exec", "sympy-1.14.0.data/data/foreign.py"):
                raw, declaration, _ = wheel("sympy", "1.14.0", additional={member: b"unapproved"})
                path = root / "unsupported.whl"
                path.write_bytes(raw)
                path.chmod(0o600)
                with self.assertRaises(ValueError):
                    worker.install_wheel(path, target, declaration, {}, time.monotonic() + 5)

    def test_real_child_positive_control_executes_pth_but_product_path_does_not(self):
        import subprocess
        import sys

        from cineforge.native_remote_package import native_package
        from cineforge.native_remote_plan import nr_json

        with tempfile.TemporaryDirectory(prefix="cf-comfy-pth-") as temporary:
            root = Path(temporary).resolve()
            site = root / "comfy-runtime/site"
            site.mkdir(parents=True, mode=0o700)
            marker = root / "executable-pth-ran"
            code = "import pathlib;pathlib.Path(" + repr(str(marker)) + ").write_text('ran')\n"
            (site / "executable.pth").write_text(code)
            control = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-S",
                    "-c",
                    "import site,sys;site.addsitedir(sys.argv[1])",
                    str(site),
                ],
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(control.returncode, 0)
            self.assertTrue(marker.exists())
            marker.unlink()
            package = root / "firstparty/cineforge"
            package.mkdir(parents=True, mode=0o700)
            for name, body in nr_json(native_package()[0]).items():
                (package / name).write_bytes(base64.b64decode(body))
            declaration = plan()
            (root / "comfy-selection.json").write_bytes(nr_encode(declaration))
            (root / "comfy-selection.json").chmod(0o600)
            # Synthetic first-party import-order fixture only: integrity/ABI have
            # separate real-file tests; no SDK/GPU/module is imported or attested.
            probe = (
                "import os,sys;sys.path.insert(0,sys.argv[1]);"
                "from cineforge import comfy_runtime_launcher as l,comfy_runtime_worker as w;"
                "l._require_abi=lambda:None;w.verify_runtime=lambda *a:None;w.verify_source=lambda *a:None;"
                "l._observe=lambda *a:dict(synthetic_first_party=True);"
                "sys.argv=['fixture',sys.argv[1],sys.argv[2],'probe',sys.argv[3]];os._exit(l.main())"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-S",
                    "-c",
                    probe,
                    str(root / "firstparty"),
                    str(root),
                    str(time.monotonic() + 5),
                ],
                capture_output=True,
                timeout=8,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(nr_json(result.stdout), {"synthetic_first_party": True})
            self.assertFalse(marker.exists())


class ComfyExecutableTests(unittest.TestCase):
    def test_only_pinned_known_elf_helpers_get_owner_execute_and_modes_are_rechecked(self):
        declaration = plan()
        header = bytearray(64)
        header[:6] = b"\x7fELF\x02\x01"
        header[16:18] = (2).to_bytes(2, "little")
        header[18:20] = (62).to_bytes(2, "little")
        body = bytes(header)  # Synthetic header, deliberately never executed.
        relative = "site/torch/bin/torch_shm_manager"
        declaration["installed"].append([relative, len(body), sha(body)])
        declaration["installed"].sort()
        declaration["executables"] = [
            {"path": relative, "size_bytes": len(body), "sha256": sha(body), "mode": 0o700}
        ]
        with tempfile.TemporaryDirectory(prefix="cf-comfy-executable-") as temporary:
            root = Path(temporary).resolve()
            path = root / "comfy-runtime" / relative
            path.parent.mkdir(parents=True, mode=0o700)
            path.write_bytes(body)
            path.chmod(0o600)
            with patch("subprocess.Popen", side_effect=AssertionError("execution")):
                worker.set_executables(declaration, root, time.monotonic() + 5)
            self.assertEqual(path.stat().st_mode & 0o7777, 0o700)
            path.write_bytes(b"wrong executable identity")
            with self.assertRaises(ValueError):
                worker.set_executables(declaration, root, time.monotonic() + 5)
        for name, bad in (
            ("mode", 0o4755),
            ("mode", True),
            ("sha256", "b" * 64),
            ("path", "site/arbitrary.py"),
        ):
            changed = copy.deepcopy(declaration)
            changed["executables"][0][name] = bad
            with self.assertRaises(ValueError):
                CheckedComfyRuntime.reviewed(changed, Reviewer())
        changed = copy.deepcopy(declaration)
        changed["executables"] = []
        with self.assertRaises(ValueError):
            CheckedComfyRuntime.reviewed(changed, Reviewer())
