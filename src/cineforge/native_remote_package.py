"""Complete fixed first-party package bytes for remote native dispatch; never ambient import."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from cineforge.native_remote_plan import nr_encode
from cineforge.private_store import read_private_input

_PACKAGE_FILES = (
    "__init__.py",
    "__main__.py",
    "audio.py",
    "audio_analysis.py",
    "audio_mix.py",
    "audio_pcm.py",
    "audio_process.py",
    "audio_review.py",
    "characters.py",
    "cli.py",
    "cloud_budget.py",
    "cloud_journal.py",
    "cloud_smoke.py",
    "comfy.py",
    "comfy_policy.py",
    "comfy_run.py",
    "comfy_transport.py",
    "comfy_runtime.py",
    "comfy_runtime_launcher.py",
    "comfy_runtime_plan.py",
    "comfy_runtime_worker.py",
    "conditioning.py",
    "conditioning_cli.py",
    "conditioning_profile.py",
    "conditioning_run.py",
    "contracts.py",
    "curated_skills.py",
    "devices.py",
    "json_codec.py",
    "mock.py",
    "native_remote.py",
    "native_remote_bootstrap.py",
    "native_remote_package.py",
    "native_remote_plan.py",
    "native_remote_worker.py",
    "native_speech.py",
    "native_speech_integrity.py",
    "native_speech_network.py",
    "native_speech_pcm.py",
    "native_speech_pins.py",
    "native_speech_process.py",
    "native_speech_worker.py",
    "paired_media.py",
    "private_store.py",
    "providers.py",
    "provisioning.py",
    "rights.py",
    "runpod.py",
    "runpod_transport.py",
    "s2v.py",
    "s2v_profile.py",
    "s2v_run.py",
    "s2v_service.py",
    "s2v_stage.py",
    "selection.py",
    "skill_runtime.py",
    "skills.py",
    "speech.py",
    "ssh_worker.py",
    "stem_import.py",
    "storage.py",
    "worker_bootstrap.py",
    "worker_bundle.py",
    "worker_guard.py",
    "worker_profiles.py",
    "workflows.py",
)


def native_package() -> tuple[bytes, tuple[tuple[str, int, str], ...]]:
    root = Path(__file__).parent
    contents = {}
    pins = []
    for name in _PACKAGE_FILES:
        raw = read_private_input(root / name, max_bytes=256 * 1024)
        contents[name] = base64.b64encode(raw).decode()
        pins.append((name, len(raw), hashlib.sha256(raw).hexdigest()))
    payload = nr_encode(contents)
    if len(payload) > 2 * 1024**2:
        raise ValueError("fixed native first party package exceeds bound")
    return payload, tuple(pins)
