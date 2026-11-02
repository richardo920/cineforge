"""Synthetic setuptools-shaped metadata; no third-party wheel install or code execution."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from cineforge.native_remote_bootstrap import nr_write
from cineforge.native_remote_worker import _install_wheel, _parent
from cineforge.native_speech_integrity import NativeSpeechError, NativeTree, digest, verify_tree
from cineforge.native_speech_pins import RUNTIME_DISTRIBUTIONS

ROOT = "setuptools-84.2.1.dist-info/"
# Published directory shapes only; bodies below are synthetic first-party fixtures.
VENDORS = (
    "autocommand-2.2.2",
    "backports.tarfile-1.2.0",
    "importlib_metadata-8.7.1",
    "jaraco.text-4.0.0",
    "jaraco_context-6.1.0",
    "jaraco_functools-4.4.0",
    "more_itertools-10.8.0",
    "packaging-26.3",
    "platformdirs-4.4.0",
    "tomli-2.4.0",
    "wheel-0.46.3",
    "zipp-3.23.0",
)


def fixture(marker=None):
    bodies = {
        ROOT + "METADATA": b"Name: setuptools\nVersion: 84.2.1\n",
        "setuptools/__init__.py": b"# synthetic inert module; never imported\n",
        ROOT + "licenses/LICENSE": b"synthetic retained notice",
    }
    for vendor in VENDORS:
        prefix = "setuptools/_vendor/" + vendor + ".dist-info/"
        # Nested metadata/RECORD are bodies covered by the owning root RECORD.
        bodies[prefix + "METADATA"] = b"Name: synthetic-vendor\nVersion: 1\n"
        bodies[prefix + "RECORD"] = b"synthetic nested record bytes; not root authority\n"
    bodies["distutils-precedence.pth"] = (
        (
            "import pathlib;pathlib.Path("
            + repr(str(marker))
            + ").write_text('synthetic hook ran')\n"
        ).encode()
        if marker is not None
        else b"# synthetic retained startup file\n"
    )
    return bodies


def wheel(path, bodies):
    bodies = dict(bodies)
    record = ROOT + "RECORD"
    rows = [
        [
            name,
            "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("="),
            str(len(raw)),
        ]
        for name, raw in bodies.items()
    ]
    rows.append([record, "", ""])
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    bodies[record] = output.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in bodies.items():
            archive.writestr(name, raw)
    path.chmod(0o600)
    return bodies


def tree(target, bodies):
    for name, raw in bodies.items():
        nr_write(_parent(target, name), raw)
    return NativeTree(
        str(target),
        tuple(
            (name, len(raw), "sha256", hashlib.sha256(raw).hexdigest())
            for name, raw in sorted(bodies.items())
        ),
    )


def full_runtime():
    bodies = fixture()
    bodies[ROOT + "RECORD"] = b"synthetic fully hashed root record"
    for name, version in RUNTIME_DISTRIBUTIONS:
        bodies[name.replace("-", "_") + "-" + version + ".dist-info/METADATA"] = (
            "Name: " + name + "\nVersion: " + version + "\n"
        ).encode()
    return bodies


class VendorMetadataTests(unittest.TestCase):
    def test_root_record_covers_all_twelve_vendor_records_code_notice_and_pth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "runtime"
            target.mkdir(mode=0o700)
            archive = root / "synthetic.whl"
            bodies = wheel(archive, fixture(root / "executed"))
            manifest = {}
            _install_wheel(
                archive,
                target,
                {"name": "setuptools", "version": "84.2.1"},
                manifest,
                time.monotonic() + 10,
            )
            self.assertEqual(set(manifest), set(bodies))
            self.assertEqual(sum(n.endswith(".dist-info/METADATA") for n in manifest), 13)
            self.assertEqual(sum(n.endswith(".dist-info/RECORD") for n in manifest), 13)
            for name, raw in bodies.items():
                self.assertEqual((target / name).read_bytes(), raw)
                self.assertEqual(manifest[name], (len(raw), hashlib.sha256(raw).hexdigest()))
            # This executes only our synthetic hook in the positive control. The
            # retained .pth is ignored by the exact production -I -B -S flags.
            marker = root / "executed"
            inactive = "import sys;sys.path.append(sys.argv[1]);import importlib.metadata"
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", inactive, str(target)],
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())
            positive = "import site,sys;site.addsitedir(sys.argv[1])"
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", positive, str(target)],
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "synthetic hook ran")

    def test_exact108_root_identity_accepts_fully_hashed_vendor_metadata_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()
            bodies = full_runtime()
            declared = tree(target, bodies)
            expected = digest(
                sorted((name, hashlib.sha256(raw).hexdigest()) for name, raw in bodies.items())
            )
            self.assertEqual(
                verify_tree(declared, time.monotonic() + 10, distributions=True), expected
            )
            vendor_record = target / "setuptools/_vendor/packaging-26.3.dist-info/RECORD"
            vendor_record.write_bytes(b"changed vendor bytes")
            with self.assertRaises(NativeSpeechError):
                verify_tree(declared, time.monotonic() + 10, distributions=True)

    def test_extra_duplicate_alias_and_ambiguous_root_identities_still_refuse(self):
        base = full_runtime()
        additions = [
            {"unknown-1.dist-info/METADATA": b"Name: unknown\nVersion: 1\n"},
            {
                "typing-extensions-4.15.0.dist-info/METADATA": b"Name: typing_extensions\nVersion: 4.15.0\n"
            },
            {"impostor-84.2.1.dist-info/METADATA": b"Name: setuptools\nVersion: 84.2.1\n"},
            {ROOT + "METADATA": b"Name: setuptools\nName: unknown\nVersion: 84.2.1\n"},
        ]
        # Derive the duplicate from the fixed lock rather than a version assumption.
        version = dict(RUNTIME_DISTRIBUTIONS)["typing-extensions"]
        additions[1] = {
            "typing-extensions-" + version + ".dist-info/METADATA": (
                "Name: typing_extensions\nVersion: " + version + "\n"
            ).encode()
        }
        for extra in additions:
            with self.subTest(paths=list(extra)), tempfile.TemporaryDirectory() as directory:
                declared = tree(Path(directory).resolve(), dict(base, **extra))
                with self.assertRaises(NativeSpeechError):
                    verify_tree(declared, time.monotonic() + 10, distributions=True)

    def test_installer_root_alias_or_extra_root_record_never_grants_vendor_authority(self):
        for extra in (
            {"other-1.dist-info/METADATA": b"Name: other\nVersion: 1\n"},
            {"other-1.dist-info/RECORD": b"not authority"},
        ):
            with self.subTest(paths=list(extra)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                target = root / "runtime"
                target.mkdir(mode=0o700)
                archive = root / "synthetic.whl"
                wheel(archive, dict(fixture(), **extra))
                with self.assertRaises(ValueError):
                    _install_wheel(
                        archive,
                        target,
                        {"name": "setuptools", "version": "84.2.1"},
                        {},
                        time.monotonic() + 10,
                    )
                self.assertEqual(list(target.iterdir()), [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "runtime"
            target.mkdir(mode=0o700)
            archive = root / "synthetic.whl"
            bodies = wheel(archive, fixture())
            with zipfile.ZipFile(archive, "w") as output:
                for name, raw in bodies.items():
                    output.writestr(name.replace(ROOT, "impostor-84.2.1.dist-info/"), raw)
            with self.assertRaises(NativeSpeechError):
                _install_wheel(
                    archive,
                    target,
                    {"name": "setuptools", "version": "84.2.1"},
                    {},
                    time.monotonic() + 10,
                )
            self.assertEqual(list(target.iterdir()), [])

    def test_vendor_record_hash_mismatch_and_unlisted_vendor_path_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "synthetic.whl"
            bodies = wheel(archive, fixture())
            bodies["setuptools/_vendor/packaging-26.3.dist-info/RECORD"] = (
                b"tampered after root hash"
            )
            with zipfile.ZipFile(archive, "w") as output:
                for name, raw in bodies.items():
                    output.writestr(name, raw)
            target = root / "runtime"
            target.mkdir(mode=0o700)
            with self.assertRaises(ValueError):
                _install_wheel(
                    archive,
                    target,
                    {"name": "setuptools", "version": "84.2.1"},
                    {},
                    time.monotonic() + 10,
                )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()
            declared = tree(target, full_runtime())
            nr_write(_parent(target, "setuptools/_vendor/unlisted.py"), b"# unexpected code")
            with self.assertRaises(NativeSpeechError):
                verify_tree(declared, time.monotonic() + 10, distributions=True)
