"""Synthetic ZIP/tar/owned-process fixtures; no actual wheel, SDK, GPU or readiness proof."""

from __future__ import annotations

import ast
import base64
import copy
import csv
import hashlib
import io
import os
import stat
import sys
import tarfile
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from cineforge import comfy_runtime_launcher as launcher
from cineforge import comfy_runtime_worker as worker
from cineforge import native_driver as driver
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
from cineforge.worker_bundle import MAX_SOURCE_BYTES, worker_source

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
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o600 << 16
            if special and path == special[0]:
                info.external_attr = special[1] << 16
            archive.writestr(info, raw)
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


class SyntheticWheelFixtureTests(unittest.TestCase):
    def test_clock_boundary_does_not_change_zip_identity_or_special_file_modes(self):
        inputs = dict(
            additional={"fixture/special": b"inert synthetic link body"},
            special=("fixture/special", stat.S_IFLNK | 0o700),
        )
        with patch.object(
            zipfile.time, "localtime", return_value=(2026, 1, 1, 0, 0, 0, 3, 1, 0)
        ) as clock:
            first, declaration, _ = wheel(**inputs)
            clock.assert_not_called()
        with patch.object(
            zipfile.time, "localtime", return_value=(2026, 1, 1, 0, 0, 2, 3, 1, 0)
        ) as clock:
            second, repeated, _ = wheel(**inputs)
            clock.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(declaration, repeated)
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertTrue(
                all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
            )
            self.assertEqual(archive.getinfo("fixture/__init__.py").external_attr >> 16, 0o600)
            self.assertEqual(
                archive.getinfo("fixture/special").external_attr >> 16, stat.S_IFLNK | 0o700
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


class ComfyVerifySourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cf-comfy-source-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "ComfyUI"
        self.source.mkdir(mode=0o700)
        (self.source / "input").mkdir(mode=0o700)
        bodies = {"main.py": b"# synthetic source\n", "input/example.png": b"example"}
        for name, body in bodies.items():
            path = self.source / name
            path.write_bytes(body)
            path.chmod(0o600)
        self.plan = {
            "source": {"files": sorted(
                [[name, len(body), sha(body)] for name, body in bodies.items()]
            )}
        }
        self.deadline = time.monotonic() + 30

    def test_server_created_input_3d_directory_passes(self):
        (self.source / "input/3d").mkdir()
        worker.verify_source(self.plan, self.root, self.deadline)

    def test_unknown_input_file_still_raises(self):
        for name in ("stray.txt", "3d"):
            with self.subTest(name=name):
                path = self.source / "input" / name
                path.write_bytes(b"stray")
                path.chmod(0o600)
                with self.assertRaisesRegex(ValueError, "unknown closed Comfy input file"):
                    worker.verify_source(self.plan, self.root, self.deadline)
                path.unlink()

    def test_file_inside_input_3d_still_raises(self):
        (self.source / "input/3d").mkdir()
        path = self.source / "input/3d/stray.txt"
        path.write_bytes(b"stray")
        path.chmod(0o600)
        with self.assertRaises(ValueError):
            worker.verify_source(self.plan, self.root, self.deadline)

    def test_hash_named_pair_file_still_passes(self):
        body = b"synthetic actor"
        path = self.source / "input" / ("actor-" + sha(body) + ".png")
        path.write_bytes(body)
        path.chmod(0o600)
        worker.verify_source(self.plan, self.root, self.deadline)


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
        # Owning root metadata is the distribution identity in both targets;
        # nested vendor metadata/RECORD/pth bytes remain fully retained data.
        path = self.root / "legacy.whl"
        path.write_bytes(raw)
        path.chmod(0o600)
        native = self.root / "native"
        native.mkdir(mode=0o700)
        native_manifest = {}
        _install_wheel(path, native, declaration, native_manifest, self.deadline)
        self.assertEqual(
            native_manifest, {name: (len(body), sha(body)) for name, body in bodies.items()}
        )
        self.assertTrue(all((native / name).read_bytes() == body for name, body in bodies.items()))

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
            spec=SimpleNamespace(image=IMAGE),
            selection=SimpleNamespace(worker_profile={"id": "m5-wan22-s2v14b-5090-v1"}),
            native_admission=gate, _require_s2v=gate
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

    def test_long_prepare_keeps_phase_cap_separate_from_setup_rpc(self):
        from test_speech import EVIDENCE

        from cineforge.s2v_service import LongSessionBudgetApproval, LongSessionServiceContext
        runtime, worker, selected, reviewer, state, bridge = self.fixture()
        worker.selection = LongSessionServiceContext(LongSessionBudgetApproval.for_scope(
            "/private/tmp/synthetic-long-prepare", "EU-RO-1", EVIDENCE, phase_id="test"))
        bridge._deadline = time.monotonic() + 50000
        bridge._expires_unix = time.time() + 50000
        original = bridge._deadline
        runtime.prepare(worker, selected, deadline=time.monotonic() + 7200)
        self.assertEqual(bridge._deadline, original)
        self.assertGreater(runtime._expires_unix, time.time() + 49000)

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
        self.assertLessEqual(len(source), MAX_SOURCE_BYTES)
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

                    def action(*, cutoff=cutoff, fixture=fixture):
                        return fixture.bridge._upload(
                            "comfy", nr_encode(plan()), comfy_expires_unix=cutoff
                        )
                else:

                    def action(*, cutoff=cutoff, fault=fault, fixture=fixture):
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
            ("source_uri", "https://synthetic-user@example.test/LICENSE"),
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
    def test_aimdo_init_precedes_downstream_lib_snapshots(self):
        for reserve_vram in (None, 1.5):
            for disable_nvml_pressure in (False, True):
                with self.subTest(reserve_vram=reserve_vram,
                                  disable_nvml_pressure=disable_nvml_pressure):
                    control = ModuleType("comfy_aimdo.control")
                    control.lib = None
                    sentinel = object()
                    calls = []

                    def init(_calls=calls, _control=control, _sentinel=sentinel, **kwargs):
                        _calls.append(kwargs)
                        _control.lib = _sentinel
                        return False  # The launcher must not gate startup on this result.

                    control.init = init
                    aimdo = ModuleType("comfy_aimdo")
                    aimdo.__path__ = []
                    aimdo.control = control
                    cli_args = SimpleNamespace(
                        args=SimpleNamespace(reserve_vram=reserve_vram,
                                             disable_nvml_pressure=disable_nvml_pressure),
                        enables_dynamic_vram=lambda: True,
                    )
                    with patch.dict(sys.modules, {
                        "comfy_aimdo": aimdo, "comfy_aimdo.control": control,
                        "comfy.cli_args": cli_args,
                    }):
                        launcher._init_aimdo()
                        snapshot = {}
                        exec("from comfy_aimdo import control\nlib = control.lib", snapshot)
                    self.assertEqual(calls, [{
                        "simple_vram_headroom": None if reserve_vram is None else int(reserve_vram * 1024 ** 3),
                        "nvml_pressure": not disable_nvml_pressure,
                    }])
                    self.assertIs(snapshot["lib"], sentinel)

    def test_aimdo_init_skipped_when_dynamic_vram_disabled(self):
        control = SimpleNamespace(lib=None, init=lambda **kwargs: None)
        cli_args = SimpleNamespace(
            args=SimpleNamespace(reserve_vram=None, disable_nvml_pressure=False),
            enables_dynamic_vram=lambda: False,
        )
        with (
            patch.dict(sys.modules, {
                "comfy_aimdo": SimpleNamespace(control=control),
                "comfy_aimdo.control": control, "comfy.cli_args": cli_args,
            }),
            patch.object(control, "init") as init,
        ):
            launcher._init_aimdo()
        init.assert_not_called()
        self.assertIsNone(control.lib)

    def test_launcher_initializes_aimdo_after_args_and_before_install(self):
        tree = ast.parse(Path(launcher.__file__).read_text())
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        aimdo_calls = [node for node in calls
                       if isinstance(node.func, ast.Name) and node.func.id == "_init_aimdo"]
        install_calls = [node for node in calls
                         if isinstance(node.func, ast.Name) and node.func.id == "install"]
        args_calls = [node for node in calls
                      if isinstance(node.func, ast.Attribute)
                      and node.func.attr == "enable_args_parsing"]
        self.assertEqual(len(aimdo_calls), 1)
        self.assertEqual(len(install_calls), 1)
        self.assertEqual(len(args_calls), 1)
        self.assertLess(args_calls[0].lineno, aimdo_calls[0].lineno)
        self.assertLess(aimdo_calls[0].lineno, install_calls[0].lineno)

    def test_probe_main_verifies_runtime_with_actual_native_session(self):
        import sys

        with tempfile.TemporaryDirectory(prefix="cf-comfy-probe-") as temporary:
            root = Path(temporary) / "comfy"
            firstparty = Path(temporary) / "native/firstparty"
            deadline = time.monotonic() + 5
            declaration = {"native": [{"driver_mount": {}}]}
            output = io.BytesIO()
            with (
                patch.object(sys, "argv", ["fixture", str(firstparty), str(root), "probe", str(deadline)]),
                patch.object(sys, "path", sys.path.copy()),
                patch.object(sys, "stdout", SimpleNamespace(buffer=output)),
                patch.object(launcher, "_require_abi"),
                patch.object(launcher, "nr_read", return_value=b"{}"),
                patch.object(launcher, "comfy_plan", return_value=declaration),
                patch.object(worker, "verify_runtime") as verify_runtime,
                patch.object(worker, "verify_source"),
                patch.object(launcher, "comfy_worker", return_value=SimpleNamespace(profile_id="fixture")),
                patch.object(launcher, "_observe", return_value={}),
                patch.object(launcher, "_init_aimdo") as init_aimdo,
            ):
                self.assertEqual(launcher.main(), 0)
            init_aimdo.assert_not_called()
            verify_runtime.assert_called_once_with(
                declaration, root, deadline, native=firstparty.parent
            )
            self.assertEqual(output.getvalue(), nr_encode({}))

    def test_driver_mount_requires_native_session_with_exact_message(self):
        binary = dict(path="/fixture/libcuda.so.1", size_bytes=1, sha256=HASH, driver_mount={})
        with self.assertRaises(ValueError) as raised:
            worker.verify_runtime_binary(binary, None, time.monotonic() + 5)
        self.assertEqual(
            str(raised.exception), "qualified NVIDIA verification needs the actual native session"
        )

    def test_import_location_requires_an_actual_nonempty_file_path(self):
        from types import SimpleNamespace

        expected = Path(__file__).resolve()
        self.assertEqual(
            launcher._module_location(SimpleNamespace(__file__=str(expected))), str(expected)
        )
        for module in (
            SimpleNamespace(),
            SimpleNamespace(__file__=None),
            SimpleNamespace(__file__=""),
            SimpleNamespace(__file__=123),
        ):
            with (
                self.subTest(module=module),
                self.assertRaisesRegex(ValueError, "location unavailable"),
            ):
                launcher._module_location(module)

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
                "l._require_abi=lambda:None;w.verify_runtime=lambda *a,**k:None;w.verify_source=lambda *a:None;"
                "l._observe=lambda *a,**k:dict(synthetic_first_party=True);"
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


class ComfyServerImageFilesTests(unittest.TestCase):
    def test_server_image_files_are_pinned_and_allowed(self):
        expected = {
            "/usr/lib/python3.11/lib-dynload/_sqlite3.cpython-311-x86_64-linux-gnu.so": (
                127752, "ef4a5bac5f08a194fe4dbb860363018962c2249f1bcf84190c3934ebc32c973a"),  # pragma: allowlist secret - public image file hash
            "/usr/lib/python3.11/lib-dynload/_zoneinfo.cpython-311-x86_64-linux-gnu.so": (
                36400, "dd15c02efc5b7bc5183550731cc21fed89b647757edb1a503861b9bf844ceec4"),  # pragma: allowlist secret - public image file hash
            "/usr/lib/python3.11/lib-dynload/resource.cpython-311-x86_64-linux-gnu.so": (
                19392, "d849af91b617c9f0e967c2cdbc1bc99f066f6afdb357a0826df01be695530a1e"),  # pragma: allowlist secret - public image file hash
            "/usr/lib/x86_64-linux-gnu/libsqlite3.so.0.8.6": (
                1468440, "eac351cf84d688d1f9235e6165bd57555b31a491cd2f65e0dca67daa69036370"),  # pragma: allowlist secret - public image file hash
            "/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12.8.4.1": (
                751771728, "10b5e6631cf8115c661eb895ed1533826308b58f7956466f53d236a40c9b622c"),  # pragma: allowlist secret - public image file hash
        }
        self.assertEqual(worker.SERVER_IMAGE_FILES, expected)
        declaration = {
            "interpreter": {"path": "/synthetic/python3.11", "size_bytes": 1, "sha256": HASH},
            "native": [], "installed": [],
        }
        with tempfile.TemporaryDirectory(prefix="cf-comfy-server-image-") as temporary:
            allowed = worker._server_allowed(declaration, Path(temporary).resolve())
            for path, pin in expected.items():
                self.assertEqual(allowed[path], pin)
            self.assertEqual(allowed["/synthetic/python3.11"], (1, HASH))
            raw_maps = "\n".join(
                f"00400000-00452000 r-xp 00000000 08:02 1234567 {path}" for path in expected
            ).encode()
            try:
                worker._verify_server_maps(allowed, raw_maps, time.monotonic() + 30)
            except ValueError as exc:
                self.assertNotIn("unreviewed native file", str(exc))
            except OSError:
                pass  # The digest-pinned Linux image files may be absent on a dev machine.

    def test_mapped_file_with_wrong_sha_still_fails_verification(self):
        with tempfile.TemporaryDirectory(prefix="cf-comfy-server-maps-") as temporary:
            path = Path(temporary).resolve() / "mapped.so"
            body = b"synthetic mapped file"
            path.write_bytes(body)
            raw_maps = f"00400000-00452000 r-xp 00000000 08:02 1234567 {path}".encode()
            allowed = {str(path): (len(body), sha(body))}
            self.assertEqual(
                worker._verify_server_maps(allowed, raw_maps, time.monotonic() + 30),
                {str(path): sha(body)},
            )
            allowed = {str(path): (
                127752, "ef4a5bac5f08a194fe4dbb860363018962c2249f1bcf84190c3934ebc32c973a")}  # pragma: allowlist secret - public image file hash
            with self.assertRaises(ValueError):
                worker._verify_server_maps(allowed, raw_maps, time.monotonic() + 30)

    def test_unknown_mapped_path_still_raises_unreviewed(self):
        declaration = {
            "interpreter": {"path": "/synthetic/python3.11", "size_bytes": 1, "sha256": HASH},
            "native": [], "installed": [],
        }
        with tempfile.TemporaryDirectory(prefix="cf-comfy-server-unknown-") as temporary:
            allowed = worker._server_allowed(declaration, Path(temporary).resolve())
            raw_maps = (
                b"00400000-00452000 r-xp 00000000 08:02 1234567 "
                b"/usr/lib/x86_64-linux-gnu/libneverreviewed.so.0"
            )
            with self.assertRaisesRegex(ValueError, "unreviewed native file"):
                worker._verify_server_maps(allowed, raw_maps, time.monotonic() + 30)


class ComfyServerDriverMapsTests(unittest.TestCase):
    def driver_session(self, root, *, tampered_receipt=False):
        path = "/usr/lib/x86_64-linux-gnu/libcuda.so.595.112.07"
        alias = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
        binding = {"worker": "synthetic-worker", "session": "synthetic-session"}
        identity = [2049, 2, 65534, stat.S_IFREG | 0o644, 1, 37, 0, 0]
        ancestors = sorted({str(parent) for candidate in (path, alias)
                            for parent in Path(candidate).parents})
        receipt = dict(
            policy=driver.DRIVER_POLICY, role="cuda_driver", alias=alias,
            binding_sha256=nr_digest(binding), identity=identity,
            user_namespace="user:[12]", mount_namespace="mnt:[13]",
            uid_map=[[0, 100000, 65536]],
            mount=[17, 1, ":".join(map(str, driver.linux_device_parts(identity[0]))),
                   path, "ext4", "ro,relatime"],
            ancestors=[[parent, [2049, index + 3, 0, stat.S_IFDIR | 0o755,
                                 2, 4096, 0, 0]]
                       for index, parent in enumerate(ancestors)],
            links=[[alias, [2049, 1, 0, stat.S_IFLNK | 0o777, 1, len(path), 0, 0], path]],
        )
        binary = dict(path=path, size_bytes=identity[5], sha256=HASH, driver_mount=receipt)
        if tampered_receipt:
            receipt["identity"][5] += 1
        native = root / "native"
        native.mkdir(mode=0o700)
        for name, value in (
            ("binding.json", binding),
            ("capabilities.json", {"binding_sha256": nr_digest(binding),
                                   "files": [dict(binary, status="measured")]}),
        ):
            (native / name).write_bytes(nr_encode(value))
            (native / name).chmod(0o600)
        declaration = {
            "interpreter": {"path": "/synthetic/python3.11", "size_bytes": 1, "sha256": HASH},
            "native": [binary], "installed": [],
        }
        return binary, native, worker._server_allowed(declaration, root), worker._server_drivers(declaration)

    def maps(self, *paths):
        return "\n".join(
            f"00400000-00452000 r-xp 00000000 08:02 1234567 {path}" for path in paths
        ).encode()

    def ordinary_file(self):
        with tempfile.NamedTemporaryFile(prefix="cf-comfy-driver-map-", dir="/tmp", delete=False) as stream:
            path = Path(stream.name).resolve()
            body = b"synthetic ordinary mapped file"
            stream.write(body)
        self.addCleanup(path.unlink)
        path.chmod(0o644)
        return str(path), (len(body), sha(body))

    def test_driver_mount_mapped_file_passes_via_receipt(self):
        with tempfile.TemporaryDirectory(prefix="cf-comfy-driver-") as temporary:
            binary, native, allowed, drivers = self.driver_session(Path(temporary).resolve())
            self.assertEqual(drivers, {binary["path"]: binary})
            raw_maps = self.maps(binary["path"])
            deadline = time.monotonic() + 30
            with patch("cineforge.native_driver.read_native_file", return_value=copy.deepcopy(binary)):
                self.assertEqual(
                    worker._verify_server_maps(allowed, raw_maps, deadline, drivers, native),
                    {binary["path"]: binary["sha256"]},
                )
                with self.assertRaisesRegex(ValueError, "needs the actual native session"):
                    worker._verify_server_maps(allowed, raw_maps, deadline, drivers, None)

    def test_driver_mount_mismatch_fails(self):
        with tempfile.TemporaryDirectory(prefix="cf-comfy-driver-") as temporary:
            binary, native, allowed, drivers = self.driver_session(Path(temporary).resolve())
            for field in ("sha256", "identity"):
                actual = copy.deepcopy(binary)
                if field == "sha256":
                    actual["sha256"] = "b" * 64
                else:
                    actual["driver_mount"]["identity"][5] += 1
                with self.subTest(live=field), patch(
                    "cineforge.native_driver.read_native_file", return_value=actual
                ), self.assertRaisesRegex(ValueError, "mounted NVIDIA binary identity changed"):
                    worker._verify_server_maps(
                        allowed, self.maps(binary["path"]), time.monotonic() + 30, drivers, native)
        with self.subTest(plan="receipt size"), tempfile.TemporaryDirectory(prefix="cf-comfy-driver-") as temporary:
            binary, native, allowed, drivers = self.driver_session(
                Path(temporary).resolve(), tampered_receipt=True)
            with patch("cineforge.native_driver.read_native_file") as read:
                with self.assertRaises(ValueError):
                    worker._verify_server_maps(
                        allowed, self.maps(binary["path"]), time.monotonic() + 30, drivers, native)
                read.assert_not_called()

    def test_foreign_uid_file_without_driver_entry_still_rejected(self):
        path, pin = self.ordinary_file()
        allowed, raw_maps, deadline = {path: pin}, self.maps(path), time.monotonic() + 30
        self.assertEqual(worker._verify_server_maps(allowed, raw_maps, deadline, {}, None),
                         {path: pin[1]})
        # Shift the accepted uid so this real file simulates a foreign-owned pod file.
        with patch.object(os, "getuid", return_value=9999):
            with self.assertRaisesRegex(ValueError, "speech staging file identity rejected"):
                worker._verify_server_maps(allowed, raw_maps, deadline, {}, None)

    def test_driver_and_ordinary_files_pass_together(self):
        with tempfile.TemporaryDirectory(prefix="cf-comfy-driver-") as temporary:
            binary, native, allowed, drivers = self.driver_session(Path(temporary).resolve())
            path, pin = self.ordinary_file()
            allowed[path] = pin
            with patch("cineforge.native_driver.read_native_file", return_value=copy.deepcopy(binary)):
                self.assertEqual(
                    worker._verify_server_maps(allowed, self.maps(binary["path"], path),
                                               time.monotonic() + 30, drivers, native),
                    {binary["path"]: binary["sha256"], path: pin[1]},
                )


class ComfyLaunchDeadlineTests(unittest.TestCase):
    def exercise(self, delay, deadline=103.0, check_log=False):
        from cineforge import worker_bootstrap
        from cineforge.worker_profiles import S2V_WORKER

        with tempfile.TemporaryDirectory(prefix="cf-comfy-launch-cap-") as temporary:
            root = Path(temporary).resolve()
            native = root / "native"
            native.mkdir(mode=0o700)
            declaration = plan()
            worker.nr_write(root / "comfy-selection.json", nr_encode(declaration))
            for index, (name, size, checksum) in enumerate(S2V_WORKER.files):
                worker.nr_write(
                    native / f"comfy-model-{index}.json",
                    nr_encode(dict(path=name, size_bytes=size, sha256=checksum)),
                )
            now, spawned = [100.0], []
            exists = Path.exists
            original_launch = launcher.launch_command

            def marker(path):
                result = exists(path)
                if path.name == delay:
                    now[0] = 104.0
                return result

            def command(*args):
                result = original_launch(*args)
                if delay == "launch-command-return" and args[-2] == "server":
                    now[0] = 104.0
                return result

            def probe(*args):
                if delay == "probe-return":
                    now[0] = 104.0
                return (
                    {
                        "python_version": "3.11",
                        "torch_version": "2.8.0+cu128",
                        "cuda_version": "12.8",
                        "gpu_memory_mib": 32768,
                    },
                    {},
                )

            def spawn(argv, **options):
                if check_log:
                    import subprocess

                    self.assertNotEqual(options["stdout"], subprocess.DEVNULL)
                    self.assertIs(options["stdout"], options["stderr"])
                    self.assertEqual(options["stdout"].name, str(root / "comfy-server.log"))
                    self.assertFalse(options["stdout"].closed)
                    self.server_log_handle = options["stdout"]
                spawned.append(
                    {
                        "at": now[0],
                        "cap": float(argv[-1]),
                        "flags": argv[1:4],
                        "jit": options["env"]["PYTORCH_JIT"],
                        "close_fds": options["close_fds"],
                    }
                )
                raise RuntimeError("blocked synthetic launch; no actual child")

            request = dict(
                plan=dict(session_id="fixture", pod_id="owned"),
                selection=dict(profile_id="m5-wan22-s2v14b-5090-v1", hardware=dict(gpu="NVIDIA GeForce RTX 5090")),
                profile=dict(id="m5-wan22-s2v14b-5090-v1", scope_sha256=HASH),
                image_digest=declaration["image"].split("@", 1)[1],
            )
            with (
                patch.object(worker_bootstrap, "worker_root", return_value=root),
                patch.object(worker, "current_comfy", return_value={}),
                patch.object(worker, "verify_source"),
                patch.object(worker_bootstrap, "bounded_process", return_value=b"{}") as child,
                patch.object(launcher, "probe_result", probe),
                patch.object(launcher, "launch_command", command),
                patch.object(worker.time, "monotonic", lambda: now[0]),
                patch.object(Path, "exists", marker),
                patch("subprocess.Popen", spawn),
            ):
                with self.assertRaises(ValueError if delay else RuntimeError):
                    worker.setup_comfy(request, native, deadline)
                child.assert_called_once()
                self.assertEqual(child.call_args.kwargs["timeout"], min(120, deadline - 100.0))
                self.assertEqual(child.call_args.kwargs["absolute_deadline"], deadline)
                before = len(spawned)
                with self.assertRaises(FileExistsError):
                    worker.setup_comfy(request, native, deadline)
                self.assertEqual(len(spawned), before)
            self.assertTrue((native / "comfy-launch-consumed").exists())
            self.assertFalse((root / "server-pid.json").exists())
            return spawned

    def test_spawn_captures_server_log_and_closes_parent_on_failure(self):
        self.assertEqual(len(self.exercise(None, check_log=True)), 1)
        self.assertTrue(self.server_log_handle.closed)

    def test_probe_allows_120_seconds_when_deadline_is_far_away(self):
        self.assertEqual(self.exercise(None, deadline=300.0)[0]["cap"], 300.0)

    def test_healthy_fixed_launch_is_consumed_and_keeps_original_child_cap(self):
        self.assertEqual(
            self.exercise(None),
            [
                {
                    "at": 100.0,
                    "cap": 103.0,
                    "flags": ["-I", "-B", "-S"],
                    "jit": "0",
                    "close_fds": True,
                }
            ],
        )

    def test_each_real_marker_read_delaying_past_cap_prevents_process_creation(self):
        for marker in ("server-pid.json", "comfy-jit-startup.json"):
            with self.subTest(marker=marker):
                self.assertEqual(self.exercise(marker), [])

    def test_late_probe_or_launch_preparation_cannot_create_production_process(self):
        for delay in ("probe-return", "launch-command-return"):
            with self.subTest(delay=delay):
                self.assertEqual(self.exercise(delay), [])


class ComfyCurrentReplyTests(unittest.TestCase):
    def test_current_comfy_reply_omits_server_log_tail_when_log_present(self):
        declaration = plan()
        hardware = dict(
            gpu_raw_name="NVIDIA GeForce RTX 5090", capability=[12, 0], device_count=1,
            nvml_total_mib=32768, nvml_reserved_mib=0, driver_version="fixture",
        )
        facts = dict(
            execution_sha256=nr_digest(declaration["execution"]),
            jit_enabled_before=False, jit_enabled_after=False, runtime=hardware,
        )
        request = dict(
            plan=dict(session_id="fixture"),
            comfy_subject_sha256=comfy_subject(declaration),
            selection=dict(profile_id="m5-wan22-s2v14b-5090-v1", hardware=dict(gpu="NVIDIA GeForce RTX 5090")),
            profile=dict(id="m5-wan22-s2v14b-5090-v1", scope_sha256=HASH),
        )
        with tempfile.TemporaryDirectory(prefix="cf-comfy-current-") as temporary:
            root = Path(temporary)
            native = root / "native"
            native.mkdir()
            for name, value in (("comfy-selection.json", declaration), ("comfy-probe.json", facts)):
                path = root / name
                path.write_bytes(nr_encode(value))
                path.chmod(0o600)
            (root / worker.COMFY_SERVER_LOG_FILENAME).write_bytes(b"synthetic server output\n")
            with (
                patch("cineforge.worker_bootstrap.worker_root", return_value=root),
                patch.object(worker, "verify_runtime"),
                patch.object(worker, "verify_source"),
                patch("cineforge.worker_bootstrap.s2v_hardware_admission", return_value=hardware),
            ):
                result = worker.current_comfy(request, native, time.monotonic() + 30)
            self.assertNotIn("server_log_tail", result)
            self.assertEqual(set(result), {
                "subject_sha256", "installed_sha256", "probe_sha256",
                "execution_sha256", "jit_enabled", "phase",
            })
