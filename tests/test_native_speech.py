"""Native boundary tests use synthetic files, fake SDK objects and owned Python children."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_speech import adapter, fixture

from cineforge import native_speech_worker as worker
from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.native_speech import NativeSpeechRuntime, OriginalVoiceDesign
from cineforge.native_speech_integrity import (
    NativeBinary,
    NativeHardware,
    NativeSpeechError,
    NativeSpeechSelection,
    NativeTree,
    digest,
    native_subject,
    verify_tree,
)
from cineforge.native_speech_network import deny_network
from cineforge.native_speech_pcm import float_pcm
from cineforge.native_speech_pins import (
    BASE_FILES,
    DESIGN_FILES,
    NETWORK_POLICY,
    PCM_POLICY,
    SDK_FILES,
)
from cineforge.native_speech_process import OwnedChildResult, pack, run_child, unpack
from cineforge.speech import SpeechError

CANARY = "SYNTHETIC-PRIVATE-NATIVE-CANARY"
HASH = "a" * 64


def pin(path, payload):
    return (path, len(payload), "sha256", hashlib.sha256(payload).hexdigest())


def selection(root):
    binary = root / "python"
    binary.write_bytes(b"synthetic interpreter; never executed")
    binary.chmod(0o600)
    python = NativeBinary(str(binary), binary.stat().st_size,
                          hashlib.sha256(binary.read_bytes()).hexdigest())
    trees = [NativeTree(str(root / name), files) for name, files in (
        ("runtime", (pin("example.dist-info/METADATA", b"Name: example\nVersion: 1\n"),)),
        ("sdk", SDK_FILES), ("base", BASE_FILES), ("design", DESIGN_FILES))]
    hardware = NativeHardware("synthetic-worker", "Synthetic CUDA", 32768, (12, 0))
    subject = native_subject(python, python, *trees, hardware)
    return NativeSpeechSelection(python, python, *trees, hardware, subject, HASH)


def response(header, payload=None):
    if payload is None:
        payload, converted = float_pcm([.1] * 2400, 24000)
    else:
        wave = inspect_pcm16_wav(payload)
        converted = SimpleNamespace(frames=wave.frames, sample_rate=wave.sample_rate,
                                    float_peak=.1, positive_endpoint_saturations=0)
    expected = header["selection"]["hardware"]
    hardware = dict(torch_version="2.8.0+cu128", torchaudio_version="2.8.0+cu128",
        cuda_version="12.8", gpu_name=expected["gpu_name"],
        memory_bytes=expected["memory_mib"] * 1024**2, capability=list(expected["capability"]),
        operator_worker_id=expected["operator_worker_id"], bf16_supported=True, attention="sdpa-math")
    metadata = dict(version=1, status="provisional", input_sha256=header["input_sha256"],
        request_sha256=digest(header["request"]), operation=header["operation"],
        seed=header["seed"], max_new_tokens=512, non_streaming_mode=True,
        selection_sha256=header["selection"]["reviewed_subject_sha256"],
        payload_sha256=hashlib.sha256(payload).hexdigest(), frames=converted.frames,
        sample_rate=converted.sample_rate, float_peak=converted.float_peak,
        positive_endpoint_saturations=converted.positive_endpoint_saturations,
        pcm_policy=PCM_POLICY, network_policy=NETWORK_POLICY,
        sdk_content_sha256=HASH, runtime_content_sha256=HASH, model_content_sha256=HASH,
        provenance="unreviewed-native-speech", hardware=hardware)
    return metadata, payload


class NativeSpeechTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cineforge-native-unit-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.selection = selection(self.root)
        self.arguments, _, _ = fixture()
        self.adapter = adapter(self.arguments)
        self.request = self.adapter.translate(utterance_id="en-line", **self.arguments)
        self.reference = inspect_pcm16_wav(self.arguments["reference_payload"])
        self.runtime = NativeSpeechRuntime(self.selection, allow_native=True, timeout_seconds=5)

    def fake_child(self, command, *, payload, env, cwd, deadline):
        self.assertEqual(command[1:5], ("-I", "-B", "-S", "-c"))
        self.assertNotIn(self.request.text, " ".join(command))
        self.assertTrue(os.path.isabs(command[0]))
        self.assertTrue(os.path.isdir(cwd))
        self.assertEqual(os.listdir(env["PATH"]), [])
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("HTTP_PROXY", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertGreater(deadline, time.monotonic())
        header, pcm = unpack(payload)
        if header["operation"] == "base":
            self.assertEqual(pcm, self.reference.pcm)
        return OwnedChildResult(pack(*response(header)), 12345)

    def test_explicit_constructor_has_no_io_and_default_adapter_stays_fake(self):
        with patch("os.open", side_effect=AssertionError("I/O")), \
                patch("subprocess.Popen", side_effect=AssertionError("process")):
            NativeSpeechRuntime(self.selection, allow_native=True)
            with self.assertRaises(NativeSpeechError):
                NativeSpeechRuntime(self.selection)
        self.assertNotIn("qwen_tts", sys.modules)
        self.assertNotIn("torch", sys.modules)
        for value in (True, 0, 2.9, float("inf"), 1801):
            with self.assertRaises(NativeSpeechError):
                NativeSpeechRuntime(self.selection, allow_native=True, timeout_seconds=value)

    def test_exact_base_and_separate_original_preview_are_provisional(self):
        with patch("cineforge.native_speech.run_child", self.fake_child):
            output = self.runtime.synthesize(self.request, self.reference)
            self.assertEqual(inspect_pcm16_wav(output).frames, 2400)
            self.assertEqual(self.runtime.last_attempt.outcome, "provisional")
            receipt = self.runtime.last_receipt
            self.assertEqual(receipt["provenance"], "unreviewed-native-speech")
            receipt["status"] = "approved"
            self.assertEqual(self.runtime.last_receipt["status"], "provisional")
            owned = self.runtime.receipt_for(self.request, output)
            self.assertEqual(owned["owned_child_status"], "exit0-parent-reaped-group-signal-attempted")
            with self.assertRaises(NativeSpeechError):
                self.runtime.receipt_for(replace(self.request, text="different"), output)
            with self.assertRaises(NativeSpeechError):
                self.runtime.receipt_for(self.request, output[:-2] + b"\1\0")
            preview = OriginalVoiceDesign("  Original line.\n", "English", "Original fictional timbre", HASH)
            self.runtime.preview(preview)
            self.assertNotEqual(self.runtime.last_receipt["input_sha256"], receipt["input_sha256"])
            self.assertEqual(self.runtime.receipt_for(self.request, output)["input_sha256"],
                             receipt["input_sha256"])
        with self.assertRaises(NativeSpeechError):
            OriginalVoiceDesign("text", "auto", "description", HASH)
        with self.assertRaises(TypeError):
            OriginalVoiceDesign("text", "English", "description", HASH, speaker="stock")

    def test_refuse_changed_reference_or_runtime_review_without_child(self):
        with patch("cineforge.native_speech.run_child") as child:
            for wave in (replace(self.reference, pcm=b"\1\0" + self.reference.pcm[2:]),
                         replace(self.reference, channels=2), replace(self.reference, sample_rate=48000)):
                with self.assertRaises(NativeSpeechError):
                    self.runtime.synthesize(self.request, wave)
            child.assert_not_called()
        with self.assertRaises(NativeSpeechError):
            replace(self.selection, reviewed_subject_sha256="b" * 64)
        with self.assertRaises(NativeSpeechError):
            replace(self.selection, sdk=NativeTree(self.selection.sdk.root, SDK_FILES[:-1]))

    def test_bad_frame_provenance_payload_and_private_failures_are_static(self):
        def child(command, **kwargs):
            header, _ = unpack(kwargs["payload"])
            meta, data = response(header)
            meta["status"] = CANARY
            return OwnedChildResult(pack(meta, data), 12345)
        for side_effect in (OSError(CANARY), SystemExit(CANARY)):
            with patch("cineforge.native_speech.run_child", side_effect=side_effect):
                self.assert_static_failure()
        with patch("cineforge.native_speech.run_child", child):
            self.assert_static_failure()
        with patch("cineforge.native_speech.run_child", return_value=OwnedChildResult(CANARY.encode(), 12345)):
            self.assert_static_failure()
        (self.root / "python").unlink()
        self.assert_static_failure()

    def assert_static_failure(self):
        caught = None
        try:
            self.runtime.synthesize(self.request, self.reference)
        except BaseException as error:
            caught = error
        self.assertIsInstance(caught, NativeSpeechError)
        self.assertNotIn(CANARY, "".join(traceback.format_exception(caught)))
        self.assertIsNone(caught.__context__)
        self.assertIsNone(caught.__cause__)
        self.assertEqual(self.runtime.last_attempt.outcome, "failed")
        self.assertIsNone(self.runtime.last_receipt)

    def test_cleanup_uncertainty_and_interruption_remain_explicit(self):
        with patch("cineforge.native_speech.run_child", side_effect=NativeSpeechError(
                CANARY, cleanup_uncertain=True, owned_process_id=12345)):
            self.assert_static_failure()
        self.assertTrue(self.runtime.last_attempt.cleanup_uncertain)
        self.assertEqual(self.runtime.last_attempt.owned_process_id, 12345)
        previous = self.runtime.last_attempt
        with patch("cineforge.native_speech.run_child") as child:
            with self.assertRaises(NativeSpeechError):
                self.runtime.synthesize(self.request, self.reference)
            with self.assertRaises(NativeSpeechError):
                self.runtime.preview(OriginalVoiceDesign("text", "English", "fictional", HASH))
            child.assert_not_called()
        self.assertIs(self.runtime.last_attempt, previous)
        # The prior error was an injected fixture, not an actual unresolved process.
        self.runtime = NativeSpeechRuntime(self.selection, allow_native=True, timeout_seconds=5)
        with patch("cineforge.native_speech.run_child", side_effect=KeyboardInterrupt(CANARY)):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.runtime.synthesize(self.request, self.reference)
        self.assertEqual(str(caught.exception), "")
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(self.runtime.last_attempt.outcome, "interrupted")

    def test_overlong_result_and_foreign_receipts_never_become_owned_success(self):
        def child(command, **kwargs):
            header, _ = unpack(kwargs["payload"])
            long, _ = float_pcm([.1] * 24000, 24000)
            return OwnedChildResult(pack(*response(header, long)), 12345)
        with patch("cineforge.native_speech.run_child", child), self.assertRaises(NativeSpeechError):
            self.runtime.synthesize(self.request, self.reference)
        self.assertEqual(self.runtime.last_attempt.outcome, "failed")
        self.assertIsNone(self.runtime.last_receipt)
        with self.assertRaises(NativeSpeechError):
            self.runtime.receipt_for(self.request, float_pcm([.1], 24000)[0])

    def test_receipt_capacity_refuses_before_spawning_and_keeps_previous_facts(self):
        with patch("cineforge.native_speech.run_child", self.fake_child):
            output = self.runtime.synthesize(self.request, self.reference)
        self.runtime._success_count = 16
        with patch("cineforge.native_speech.run_child") as child:
            with self.assertRaises(NativeSpeechError):
                self.runtime.synthesize(self.request, self.reference)
            child.assert_not_called()
        self.assertEqual(self.runtime.receipt_for(self.request, output)["status"], "provisional")

    def test_existing_adapter_rechecks_revocation_during_native_call(self):
        def child(command, **kwargs):
            self.arguments["ledger"].revoke(self.arguments["reference"].permission.rights,
                                            reason="synthetic withdrawal")
            return self.fake_child(command, **kwargs)
        instance = adapter(self.arguments, runtime=self.runtime)
        with patch("cineforge.native_speech.run_child", child), self.assertRaises(SpeechError):
            instance.synthesize(utterance_id="en-line", output_id="out", output_revision=1,
                                **self.arguments)

    def test_worker_policy_and_integrity_precede_sdk_calls(self):
        request = self.request.to_dict()
        header = {"version": 1, "operation": "base", "request": request,
            "selection": asdict(self.selection), "seed": 0, "deadline": time.monotonic() + 20,
            "input_sha256": digest({"operation": "base", "request": request, "seed": 0,
                "selection": self.selection.subject_sha256(),
                "reference_pcm_sha256": hashlib.sha256(self.reference.pcm).hexdigest()})}
        header, _ = unpack(pack(header, b""))
        order = []
        def policy(*args):
            order.append("deny")
            return NETWORK_POLICY
        def verified(*args, **kwargs):
            self.assertEqual(order[0], "deny")
            order.append("verify")
            return HASH
        def generate(*args):
            self.assertEqual(order, ["deny", "verify", "verify", "verify"])
            order.append("sdk")
            return (*float_pcm([.1] * 2400, 24000), response(header)[0]["hardware"])
        with patch.object(worker, "deny_network", policy), patch.object(worker, "verify_tree", verified), \
                patch.object(worker, "_generate", generate):
            meta, _ = worker.handle(header, self.reference.pcm)
        self.assertEqual(meta["provenance"], "unreviewed-native-speech")
        with patch.object(worker, "deny_network", side_effect=NativeSpeechError("unavailable")), \
                patch.object(worker, "_generate") as sdk:
            with self.assertRaises(NativeSpeechError):
                worker.handle(header, self.reference.pcm)
            sdk.assert_not_called()


class NativeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cineforge-native-integrity-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.payload = b'{"model_type":"synthetic"}'
        self.path = self.root / "config.json"
        self.path.write_bytes(self.payload)
        self.path.chmod(0o600)
        self.tree = NativeTree(str(self.root), (pin("config.json", self.payload),))

    def verify(self):
        return verify_tree(self.tree, time.monotonic() + 5)

    def test_exact_files_and_missing_tampered_extra_or_linked_refuse(self):
        self.assertEqual(len(self.verify()), 64)
        self.path.write_bytes(b"tampered")
        with self.assertRaises(NativeSpeechError):
            self.verify()
        self.path.write_bytes(self.payload)
        extra = self.root / "extra.py"
        extra.write_bytes(b"unreviewed")
        with self.assertRaises(NativeSpeechError):
            self.verify()
        extra.unlink()
        os.link(self.path, extra)
        with self.assertRaises(NativeSpeechError):
            self.verify()
        extra.unlink()
        self.path.rename(extra)
        self.path.symlink_to(extra)
        with self.assertRaises((NativeSpeechError, OSError)):
            self.verify()

    def test_runtime_metadata_closure_and_remote_config_are_not_authority(self):
        self.path.unlink()
        directory = self.root / "fixture.dist-info"
        directory.mkdir(mode=0o700)
        payload = b"Name: fixture\nVersion: 1\n"
        (directory / "METADATA").write_bytes(payload)
        self.tree = NativeTree(str(self.root), (pin("fixture.dist-info/METADATA", payload),))
        with patch("cineforge.native_speech_integrity.RUNTIME_DISTRIBUTIONS", (("fixture", "1"),)):
            self.assertEqual(len(verify_tree(self.tree, time.monotonic() + 5, distributions=True)), 64)
        with self.assertRaises(NativeSpeechError):
            verify_tree(self.tree, time.monotonic() + 5, distributions=True)
        (directory / "METADATA").unlink()
        directory.rmdir()
        for name in ("auto_map", "_attn_implementation_internal", "code_revision"):
            value = json.dumps({"nested": {name: "remote/repo"}}).encode()
            self.path.write_bytes(value)
            self.tree = NativeTree(str(self.root), (pin("config.json", value),))
            with self.assertRaises(NativeSpeechError):
                self.verify()

    def test_hashing_observes_original_deadline_and_private_directory(self):
        with self.assertRaises(NativeSpeechError):
            verify_tree(self.tree, time.monotonic() - 1)
        self.root.chmod(0o755)
        with self.assertRaises(NativeSpeechError):
            self.verify()
        for name in ("../escape", "/absolute", "a//b", "a/./b"):
            with self.assertRaises(NativeSpeechError):
                NativeTree(str(self.root), (pin(name, b"x"),))

    def test_unavailable_network_policy_sanitizes_without_sdk_import(self):
        native = NativeBinary(str(self.path), len(self.payload), hashlib.sha256(self.payload).hexdigest())
        with patch("cineforge.native_speech_network.sys.platform", "unsupported"):
            try:
                deny_network(native, time.monotonic() + 5)
            except NativeSpeechError as error:
                self.assertIsNone(error.__context__)
            else:
                self.fail("missing isolation accepted")


class NativePCMTests(unittest.TestCase):
    def test_unit_range_ties_even_and_positive_endpoint_are_explicit(self):
        result, meta = float_pcm([-1., 1., .5 / 32768, 1.5 / 32768, -.5 / 32768], 24000)
        self.assertEqual(struct.unpack("<5h", inspect_pcm16_wav(result).pcm), (-32768, 32767, 0, 2, 0))
        self.assertEqual(meta.positive_endpoint_saturations, 1)
        self.assertEqual(meta.float_peak, 1)
        for samples in ([float("nan")], [float("inf")], [1.001], [-1.001], [True], [1], [], [[.1]]):
            with self.assertRaises(NativeSpeechError):
                float_pcm(samples, 24000)
        with self.assertRaises(NativeSpeechError):
            float_pcm([.1], True)


class NativeSDKBridgeTests(unittest.TestCase):
    def test_fixed_source_supported_arguments_with_fake_sdk_and_no_site_hooks(self):
        arguments, _, _ = fixture()
        translated = adapter(arguments)
        calls = []

        class Array:
            ndim = 1
            dtype = "float32"

            def __init__(self, values):
                self.values = values
                self.size = len(values)

            def astype(self, dtype):
                calls.append(("reference-dtype", dtype))
                return self

            def __truediv__(self, value):
                calls.append(("reference-scale", value))
                return Array([float(item) / value for item in self.values])

            def tolist(self):
                return self.values

        def frombuffer(data, dtype):
            self.assertEqual(dtype, "<i2")
            self.assertEqual(data, inspect_pcm16_wav(arguments["reference_payload"]).pcm)
            return Array(list(struct.unpack("<" + "h" * (len(data) // 2), data)))

        class Model:
            cpu = False

            def __init__(self, kind):
                parameter = SimpleNamespace(device=SimpleNamespace(type="cpu" if self.cpu else "cuda",
                                                                   index=None if self.cpu else 0))
                def parameters():
                    return iter((parameter,))
                self.model = SimpleNamespace(tts_model_type=kind,
                    tokenizer_type="qwen3_tts_tokenizer_12hz", parameters=parameters,
                    speech_tokenizer=SimpleNamespace(model=SimpleNamespace(parameters=parameters)))

            def create_voice_clone_prompt(self, **kwargs):
                calls.append(("prompt", kwargs))
                return "fixed-prompt-items"

            def generate_voice_clone(self, **kwargs):
                calls.append(("clone", kwargs))
                return [Array([.1] * 2400)], 24000

            def generate_voice_design(self, **kwargs):
                calls.append(("design", kwargs))
                return [Array([.1] * 2400)], 24000

        def load(path, **kwargs):
            calls.append(("load", path, kwargs))
            return Model("base" if path.endswith("base") else "voice_design")

        numpy = SimpleNamespace(frombuffer=frombuffer, float32="float32", ndarray=Array,
                                dtype=lambda value: value)
        torch = SimpleNamespace(__version__="2.8.0+cu128", version=SimpleNamespace(cuda="12.8"),
            bfloat16="bf16", cuda=SimpleNamespace(is_available=lambda: True,
                device_count=lambda: 1, is_bf16_supported=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(name="Synthetic CUDA",
                    total_memory=32768 * 1024**2), get_device_capability=lambda index: (12, 0)),
            manual_seed=lambda value: calls.append(("seed", value)),
            inference_mode=contextlib.nullcontext, backends=SimpleNamespace(cuda=SimpleNamespace(
                enable_flash_sdp=lambda value: calls.append(("flash", value)),
                enable_mem_efficient_sdp=lambda value: calls.append(("efficient", value)),
                enable_cudnn_sdp=lambda value: calls.append(("cudnn", value)),
                enable_math_sdp=lambda value: calls.append(("math", value)))))
        modules = {"numpy": numpy, "torch": torch,
                   "torchaudio": SimpleNamespace(__version__="2.8.0+cu128"),
                   "qwen_tts": SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load))}
        with tempfile.TemporaryDirectory(prefix="cineforge-sdk-fake-") as temporary:
            chosen = selection(Path(temporary).resolve())
            with patch("site.addsitedir", side_effect=AssertionError("site hook execution")), \
                    patch.object(worker.importlib, "import_module", side_effect=lambda name: modules[name]), \
                    patch.object(worker.sys, "path", sys.path.copy()):
                for identity, language in (("en-line", "English"), ("zh-line", "Chinese")):
                    request = translated.translate(utterance_id=identity, **arguments)
                    output, _, observed = worker._generate("base", request,
                        inspect_pcm16_wav(arguments["reference_payload"]).pcm, chosen, 17)
                    self.assertEqual(observed["gpu_name"], "Synthetic CUDA")
                    self.assertEqual(inspect_pcm16_wav(output).frames, 2400)
                    clone = next(row[1] for row in reversed(calls) if row[0] == "clone")
                    self.assertEqual(clone, dict(text=request.text, language=language,
                        voice_clone_prompt="fixed-prompt-items", non_streaming_mode=True,
                        max_new_tokens=512))
                    prompt = next(row[1] for row in reversed(calls) if row[0] == "prompt")
                    self.assertEqual(prompt["ref_text"], request.ref_text)
                    self.assertIs(prompt["x_vector_only_mode"], False)
                    self.assertEqual(prompt["ref_audio"][1], request.reference_sample_rate)
                design = OriginalVoiceDesign("Original reference", "Chinese", "Fictional timbre", HASH)
                worker._generate("design", design, b"", chosen, 18)
                loads = sum(row[0] == "load" for row in calls)
                request = translated.translate(utterance_id="en-line", **arguments)
                reference = inspect_pcm16_wav(arguments["reference_payload"]).pcm
                for attribute, value in (("is_available", lambda: False),
                                         ("is_bf16_supported", lambda: False),
                                         ("get_device_capability", lambda index: (7, 5)),
                                         ("get_device_properties", lambda index:
                                          SimpleNamespace(name="Other GPU", total_memory=32768 * 1024**2))):
                    with patch.object(torch.cuda, attribute, value), self.assertRaises(NativeSpeechError):
                        worker._generate("base", request, reference, chosen, 1)
                self.assertEqual(sum(row[0] == "load" for row in calls), loads)
                with patch.object(Array, "ndim", 2), self.assertRaises(NativeSpeechError):
                    worker._generate("base", request, reference, chosen, 1)
                with patch.object(Model, "cpu", True), self.assertRaises(NativeSpeechError):
                    worker._generate("base", request, reference, chosen, 1)
        for row in calls:
            if row[0] == "load":
                self.assertEqual(row[2], dict(local_files_only=True, trust_remote_code=False,
                    use_safetensors=True, device_map="cuda:0", dtype="bf16", attn_implementation="sdpa"))
        self.assertEqual(next(row[1] for row in reversed(calls) if row[0] == "design"),
            dict(text=design.text, language=design.language, instruct=design.description,
                 non_streaming_mode=True, max_new_tokens=512))
        self.assertIn(("reference-scale", 32768), calls)
        for option in ("flash", "efficient", "cudnn"):
            self.assertIn((option, False), calls)
        self.assertIn(("math", True), calls)


class NativeProcessTests(unittest.TestCase):
    def test_descriptor_reuse_is_distinguished_from_an_aliased_open_file(self):
        from tests_linux_native.test_native_speech_network import descriptor_identity, same_identity
        with tempfile.TemporaryDirectory(prefix="cineforge-fd-identity-") as temporary:
            root = Path(temporary)
            first = os.open(root / "first", os.O_CREAT | os.O_RDWR, 0o600)
            alias = os.dup(first)
            second = os.open(root / "second", os.O_CREAT | os.O_RDWR, 0o600)
            try:
                expected = descriptor_identity(first)
                self.assertTrue(same_identity(descriptor_identity(alias), expected))
                os.dup2(second, first)
                self.assertEqual(descriptor_identity(first)["fd"], expected["fd"])
                self.assertFalse(same_identity(descriptor_identity(first), expected))
                self.assertTrue(same_identity(descriptor_identity(alias), expected))
            finally:
                for descriptor in (first, alias, second):
                    os.close(descriptor)

    def execute(self, code, *, seconds=4):
        return run_child((sys.executable, "-I", "-S", "-c", code), payload=b"fixture",
            env={"PATH": "/usr/bin:/bin", "LANG": "C"}, cwd="/", deadline=time.monotonic() + seconds)

    def test_owned_standin_child_io_and_failure_redaction(self):
        code = "import os,sys;sys.stdin.buffer.read();os.write(1,b'fixture');os._exit(0)"
        try:
            output = self.execute(code)
        except NativeSpeechError as error:
            # This host's Darwin kernel refuses killpg on a dead owned group.
            # That is explicit uncertainty, not a claimed macOS native success.
            self.assertEqual(sys.platform, "darwin")
            self.assertTrue(error.cleanup_uncertain)
        else:
            self.assertEqual(output.payload, b"fixture")
        with self.assertRaises(NativeSpeechError) as caught:
            self.execute("import sys;sys.stdin.buffer.read();raise RuntimeError('" + CANARY + "')")
        self.assertNotIn(CANARY, "".join(traceback.format_exception(caught.exception)))

    def test_timeout_output_bound_and_interruption_reap_owned_child(self):
        original = subprocess.Popen
        children = []
        def popen(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            return process
        start = time.monotonic()
        with patch("cineforge.native_speech_process.subprocess.Popen", popen):
            with self.assertRaises(NativeSpeechError):
                self.execute("import time;time.sleep(10)", seconds=3)
            with patch("cineforge.native_speech_process.MAX_OUTPUT", 32), \
                    self.assertRaises(NativeSpeechError):
                self.execute("import os,sys;sys.stdin.buffer.read();os.write(1,b'x'*1000)")
            with patch("selectors.EpollSelector.select" if sys.platform == "linux"
                       else "selectors.KqueueSelector.select", side_effect=KeyboardInterrupt(CANARY)), \
                    self.assertRaises(KeyboardInterrupt):
                self.execute("import time;time.sleep(10)")
        self.assertLess(time.monotonic() - start, 5)
        self.assertTrue(all(process.returncode is not None for process in children))

    def test_isolated_no_site_startup_ignores_synthetic_user_hook(self):
        with tempfile.TemporaryDirectory(prefix="cineforge-site-hook-") as temporary:
            root = Path(temporary).resolve()
            marker = root / "executed"
            hook = "from pathlib import Path;Path(" + repr(str(marker)) + ").write_text('bad')\n"
            (root / "sitecustomize.py").write_text(hook)
            site = root / "lib" / ("python" + sys.version.split()[0].rsplit(".", 1)[0]) / "site-packages"
            site.mkdir(parents=True)
            (site / "unreviewed.pth").write_text(hook)
            package = str(Path(__file__).resolve().parents[1] / "src")
            from cineforge.native_speech import _BOOTSTRAP
            result = subprocess.run((sys.executable, "-I", "-B", "-S", "-c", _BOOTSTRAP, package),
                input=b"invalid", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(root), "PYTHONUSERBASE": str(root)})
            self.assertEqual(result.returncode, 1)
            self.assertEqual(unpack(result.stdout), ({"version": 1, "status": "failed"}, b""))
            self.assertFalse(marker.exists())
            self.assertEqual(result.stderr, b"")

    def test_child_discards_sdk_file_descriptor_output_and_private_tracebacks(self):
        package = str(Path(__file__).resolve().parents[1] / "src")
        code = ("import os,sys;sys.path.insert(0,sys.argv[1]);"
                "from cineforge import native_speech_worker as w;"
                "exec(\"def fail(*args):\\n os.write(1,b'" + CANARY + "');"
                "os.write(2,b'" + CANARY + "');raise RuntimeError('" + CANARY + "')\\n\");"
                "w.handle=fail;os._exit(w.main())")
        result = subprocess.run([sys.executable, "-I", "-B", "-S", "-c", code, package],
            input=pack({}, b""), capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(unpack(result.stdout), ({"version": 1, "status": "failed"}, b""))
        self.assertEqual(result.stderr, b"")
        self.assertNotIn(CANARY.encode(), result.stdout)

    def test_cleanup_attempts_all_steps_despite_private_faults(self):
        from cineforge.native_speech_process import _finish
        calls = []
        def failure(name):
            def call(*args, **kwargs):
                calls.append(name)
                raise OSError(CANARY)
            return call
        process = SimpleNamespace(pid=123, kill=failure("fallback"), wait=failure("wait"),
            stdin=SimpleNamespace(close=failure("stdin")), stdout=SimpleNamespace(close=failure("stdout")))
        selector = SimpleNamespace(close=failure("selector"))
        with patch("os.killpg", failure("group")):
            self.assertTrue(_finish(process, selector, time.monotonic() + 3))
        self.assertEqual(calls, ["group", "fallback", "wait", "selector", "stdin", "stdout"])


if __name__ == "__main__":
    unittest.main()
