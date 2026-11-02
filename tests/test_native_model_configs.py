"""Model roles use synthetic exact-shaped pins; SDK/runtime resources remain opaque."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cineforge import native_remote_worker as stage
from cineforge import native_speech_integrity as integrity
from cineforge.native_remote_bootstrap import nr_write
from cineforge.native_remote_plan import nr_encode
from cineforge.native_remote_worker import _parent
from cineforge.native_speech_integrity import NativeSpeechError, NativeTree, digest, verify_tree
from cineforge.native_speech_pins import BASE_FILES, DESIGN_FILES, RUNTIME_DISTRIBUTIONS

CONFIGS = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json",
    "tokenizer_config.json",
}


def model_bodies(role):
    pins = BASE_FILES if role == "base" else DESIGN_FILES
    return {
        p[0]: json.dumps({"model_type": "synthetic-" + role, "path": p[0]}).encode()
        if p[0] in CONFIGS
        else b"synthetic inert " + role.encode()
        for p in pins
    }


def declared(root, bodies):
    return NativeTree(
        str(root),
        tuple(
            (p, len(raw), "sha256", hashlib.sha256(raw).hexdigest()) for p, raw in bodies.items()
        ),
    )


def write_tree(root, bodies):
    for p, raw in bodies.items():
        nr_write(_parent(root, p), raw)
    return declared(root, bodies)


class ModelRoleTests(unittest.TestCase):
    def test_fixed_stage_routes_sdk_runtime_and_each_model_to_its_intended_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for role in ("package", "recipe", "sox"):
                nr_write(root / (role + ".blob.done"), b"synthetic complete")
            nr_write(root / "recipe.blob", b"{}")
            nr_write(root / "sox.blob", b"x")
            binary = {"path": str(root / "synthetic-python"), "size_bytes": 1, "sha256": "a" * 64}
            nr_write(root / "preflight.json", nr_encode({"interpreter": binary, "seccomp": binary}))
            recipe = {
                "reviewed_subject_sha256": "a" * 64,
                "wheels": [
                    {"name": "sox", "size_bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
                ],
            }
            calls = {}

            def inspect(tree, deadline, **role):
                calls[Path(tree.root).name] = role
                return "a" * 64

            def synthetic_install(path, runtime, wheel, manifest, deadline):
                manifest["sox-1.5.0.dist-info/METADATA"] = (1, "a" * 64)

            with (
                patch.object(stage, "nr_recipe", return_value=recipe),
                patch.object(stage, "verify_binary"),
                patch.object(stage, "_download"),
                patch.object(stage, "_install_wheel", synthetic_install),
                patch.object(stage, "verify_tree", inspect),
            ):
                stage._setup(root, time.monotonic() + 10)
            self.assertEqual(
                calls,
                {
                    "sdk": {"distributions": False, "model_role": None},
                    "runtime": {"distributions": True, "model_role": None},
                    "base": {"distributions": False, "model_role": "base"},
                    "design": {"distributions": False, "model_role": "design"},
                },
            )

    def test_explicit_seven_paths_are_actual_curated_base_and_design_configs(self):
        for pins in (BASE_FILES, DESIGN_FILES):
            actual = {p[0] for p in pins if p[0].endswith(("config.json", "configuration.json"))}
            self.assertEqual(actual, CONFIGS)
        self.assertEqual(integrity._MODEL_CONFIG_PATHS, CONFIGS)

    def test_runtime_jsonc_and_forbidden_keys_are_hashed_data_not_model_authority(self):
        bodies = {
            name.replace("-", "_") + "-" + version + ".dist-info/METADATA": (
                "Name: " + name + "\nVersion: " + version + "\n"
            ).encode()
            for name, version in RUNTIME_DISTRIBUTIONS
        }
        bodies.update(
            {
                "gradio/_frontend_code/client/tsconfig.json": b'{/* Bundler */"ok":true}',
                "runtime/source/config.json": b'{"auto_map":{"trust_remote_code":true}}',
                "sdk/configuration.json": b'{"nested":[{"code_revision":"data"}]}',
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            tree = write_tree(root, bodies)
            expected = digest(
                sorted((p, hashlib.sha256(raw).hexdigest()) for p, raw in bodies.items())
            )
            self.assertEqual(verify_tree(tree, time.monotonic() + 10, distributions=True), expected)
            (root / "gradio/_frontend_code/client/tsconfig.json").write_bytes(b"tampered data")
            with self.assertRaises(NativeSpeechError):
                verify_tree(tree, time.monotonic() + 10, distributions=True)

    def test_fixed_role_and_full_pins_are_required_before_any_model_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base = write_tree(root, model_bodies("base"))
            design = declared(root, model_bodies("design"))
            with (
                patch.object(integrity, "BASE_FILES", base.files),
                patch.object(integrity, "DESIGN_FILES", design.files),
            ):
                self.assertEqual(
                    len(verify_tree(base, time.monotonic() + 10, model_role="base")), 64
                )
                for role, distro in (
                    (None, False),
                    ("design", False),
                    ("base", True),
                    ("other", False),
                    (True, False),
                ):
                    with self.assertRaises(NativeSpeechError):
                        verify_tree(
                            base, time.monotonic() + 10, model_role=role, distributions=distro
                        )
                for pins in (
                    base.files[:-1],
                    base.files + (("checkpoint.pkl", 1, "sha256", "a" * 64),),
                ):
                    with self.assertRaises(NativeSpeechError):
                        verify_tree(
                            replace(base, files=pins), time.monotonic() + 10, model_role="base"
                        )

    def test_all_model_config_paths_reject_recursive_remote_controls_for_both_roles(self):
        for role in ("base", "design"):
            for path in CONFIGS:
                for forbidden in (
                    "auto_map",
                    "trust_remote_code",
                    "_attn_implementation_internal",
                    "custom_pipelines",
                    "code_revision",
                ):
                    with (
                        self.subTest(role=role, path=path, field=forbidden),
                        tempfile.TemporaryDirectory() as directory,
                    ):
                        root = Path(directory).resolve()
                        bodies = model_bodies(role)
                        bodies[path] = json.dumps(
                            {"nested": [{"deep": {forbidden: False}}]}
                        ).encode()
                        tree = write_tree(root, bodies)
                        with patch.object(
                            integrity,
                            "BASE_FILES" if role == "base" else "DESIGN_FILES",
                            tree.files,
                        ):
                            with self.assertRaises(NativeSpeechError):
                                verify_tree(tree, time.monotonic() + 10, model_role=role)

    def test_both_roles_keep_strict_json_object_and_exact_file_identity_checks(self):
        invalid = (
            b"{/* no JSONC models */}",
            b"[]",
            b"null",
            b"1",
            b'"config"',
            b"NaN",
            b'{"x":Infinity}',
            b'{"x":1e999}',
            b'{"x":1,"x":2}',
            b"\xff",
        )
        for role in ("base", "design"):
            for payload in invalid:
                with (
                    self.subTest(role=role, payload=payload),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory).resolve()
                    bodies = model_bodies(role)
                    bodies["speech_tokenizer/configuration.json"] = payload
                    tree = write_tree(root, bodies)
                    with patch.object(
                        integrity, "BASE_FILES" if role == "base" else "DESIGN_FILES", tree.files
                    ):
                        with self.assertRaises(NativeSpeechError):
                            verify_tree(tree, time.monotonic() + 10, model_role=role)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                bodies = model_bodies(role)
                tree = write_tree(root, bodies)
                with patch.object(
                    integrity, "BASE_FILES" if role == "base" else "DESIGN_FILES", tree.files
                ):
                    self.assertEqual(
                        len(verify_tree(tree, time.monotonic() + 10, model_role=role)), 64
                    )
                    (root / "config.json").write_bytes(b'{"model_type":"changed"}')
                    with self.assertRaises(NativeSpeechError):
                        verify_tree(tree, time.monotonic() + 10, model_role=role)
