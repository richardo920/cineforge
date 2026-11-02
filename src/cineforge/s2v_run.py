"""Bounded one-attempt S2V execution on an already owned, guarded, staged worker.

Never acquires, installs, stages speech synthesis or releases the caller-owned lease.
Video is silent: exact source speech remains paired by identity for later mixing/review.
"""
from __future__ import annotations

import hashlib
import math
import shutil
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, cast

from cineforge import s2v_profile as profile
from cineforge.comfy_policy import WorkerVerification
from cineforge.comfy_run import execute_comfy
from cineforge.contracts import RunManifest
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.s2v import S2VError, S2VSelection
from cineforge.s2v_service import S2VServiceContext
from cineforge.ssh_worker import SSHWorker, run_process
from cineforge.worker_bootstrap import validate_s2v_runtime
from cineforge.workflows import canonical, strict_object


def inspect_s2v_media(path: Path) -> dict[str, Any]:
    """Fixed local Matroska/VP9 probe; no network protocols or arbitrary media URL."""
    result = None
    try:
        binary = shutil.which("ffprobe")
        if binary is None:
            raise ValueError
        data = strict_object(run_process([binary, "-v", "error", "-max_alloc", "67108864",
            "-protocol_whitelist", "file,pipe", "-format_whitelist", "matroska,webm",
            "-f", "matroska", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames:format=duration",
            "-of", "json", str(path)], timeout=10, max_output=65536))
        streams = data["streams"]
        if not isinstance(streams, list) or len(streams) != 1:
            raise ValueError
        stream = streams[0]
        if (stream["codec_type"] != "video" or stream["codec_name"] != "vp9" or stream["pix_fmt"] != "yuv420p"
                or type(stream["width"]) is not int or stream["width"] != profile.WIDTH
                or type(stream["height"]) is not int or stream["height"] != profile.HEIGHT
                or type(stream["nb_read_frames"]) is not str or stream["nb_read_frames"] != "77"
                or type(stream["avg_frame_rate"]) is not str or len(stream["avg_frame_rate"]) > 32
                or Fraction(stream["avg_frame_rate"]) != 16):
            raise ValueError
        duration = float(data["format"]["duration"])
        if not math.isfinite(duration) or abs(duration - profile.DURATION) > 1 / profile.FPS:
            raise ValueError
        result = {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
            "width": profile.WIDTH, "height": profile.HEIGHT, "decoded_frames": profile.FRAMES,
            "fps": float(profile.FPS), "duration_seconds": duration}
    except (Exception, KeyboardInterrupt):
        pass
    if result is None:
        raise S2VError("bounded local S2V video inspection unavailable or rejected")
    return result


def technical_result(selection: S2VSelection, manifest: RunManifest, directory: Path,
                     inspect_media: Callable[[Path], dict[str, Any]] = inspect_s2v_media) -> dict[str, Any]:
    """Actual decoded output plus paired speech timing, not semantic or visible-sync review."""
    result = None
    try:
        selection.gate.preflight()
        if (len(manifest.artifacts) != 1 or len(manifest.shot_runs) != 1
                or manifest.shot_runs[0].status.value != "succeeded"
                or manifest.plan != selection.plan()):
            raise ValueError
        artifact = manifest.artifacts[0]
        if artifact.uri != "artifact-" + artifact.sha256 + ".webm":
            raise ValueError
        path = directory / artifact.uri
        before = read_private_input(path, max_bytes=32 * 1024 * 1024)
        if (len(before) != artifact.size_bytes
                or hashlib.sha256(before).hexdigest() != artifact.sha256):
            raise ValueError
        media = inspect_media(path)
        expected = {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
                    "width": profile.WIDTH, "height": profile.HEIGHT,
                    "decoded_frames": profile.FRAMES, "fps": float(profile.FPS)}
        duration = media.get("duration_seconds")
        if (set(media) != set(expected) | {"duration_seconds"}
                or any(type(media.get(key)) is bool or media.get(key) != value
                       for key, value in expected.items())
                or any(type(media.get(key)) is not int for key in ("width", "height", "decoded_frames"))
                or type(duration) not in (int, float) or not math.isfinite(cast(float, duration))
                or abs(cast(float, duration) - profile.DURATION) > 1 / profile.FPS
                or read_private_input(path, max_bytes=32 * 1024 * 1024) != before):
            raise ValueError
        selection.gate.preflight()
        speech = selection.gate.speech
        result = {"video": media, "video_sha256": artifact.sha256,
            "speech": speech.pin.to_dict(), "speech_frames": speech.frames,
            "speech_sample_rate": speech.sample_rate, "speech_duration_seconds": speech.frames / speech.sample_rate,
            "picture_duration_seconds": profile.DURATION, "start_offset_seconds": 0,
            "picture_duration_tolerance_seconds": 1 / profile.FPS,
            "conditioning_timing": "pinned core50Hz->30Hz->16fps; zero feature buckets beyond speech",
            "speech_modification": "none", "soundtrack": "paired separately; video is silent",
            "visible_lip_sync": "not_assessed", "identity_review": "pending_human_review",
            "language_intelligibility": "not_assessed", "export_authority": "none"}
    except (Exception, KeyboardInterrupt):
        pass
    if result is None:
        raise S2VError("actual S2V video or paired speech timing contract rejected")
    return result


def _admit(selection: S2VSelection, verification: WorkerVerification, driver: SSHWorker) -> None:
    if (type(selection) is not S2VSelection or type(verification) is not WorkerVerification
            or type(driver.selection) is not S2VServiceContext
            or driver.selection != selection.gate.service or driver.plan is None
            or driver.s2v_verification != verification
            or driver.plan.pod_id != verification.record["worker_id"]
            or driver.runtime_evidence is None):
        raise S2VError("S2V execution requires its exact prepared guarded worker")
    validate_s2v_runtime(driver.runtime_evidence)
    selection.require_execution_worker(verification)
    driver.check()
    driver.verify_comfy_current()


def run_s2v(selection: S2VSelection, verification: WorkerVerification, driver: SSHWorker,
            output_dir: str | Path, *, allow_live: bool = False,
            execute: Callable[..., RunManifest] = execute_comfy,
            inspect_media: Callable[[Path], dict[str, Any]] = inspect_s2v_media,
            clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """No retries. Caller keeps outer budget/guard/enforcer and guaranteed lease cleanup.

    Requires already checked paired staging; no private workflow reaches an unproven
    forward. Existing execute_comfy owns atomic cancellation and bounded result reserve.
    Any attempted run consumes this nonserialized intent, including lost acknowledgements.
    """
    record = None
    try:
        if allow_live is not True:
            raise ValueError
        _admit(selection, verification, driver)
        remaining = driver.budget.remaining_seconds(driver._key)
        if not math.isfinite(remaining) or remaining <= 150:
            raise ValueError
        timeout = min(600, remaining - 120)
        deadline = clock() + timeout
        trace = selection.gate.preflight()
        record = {"schema_version": 1, "mode": "m5-s2v-one-shot", "outcome": "prepared",
            "subject_sha256": selection.approved_subject_sha256,
            "scope_sha256": selection.gate.service.approval.subject_sha256,
            "lineage": trace, "worker_verification": verification.record,
            "worker_source_sha256": hashlib.sha256(driver._source).hexdigest(),
            "controller_source_sha256": hashlib.sha256(read_private_input(
                Path(__file__), max_bytes=128 * 1024)).hexdigest(),
            "runner_revision": profile.RUNNER_REVISION,
            "shot_record": "shot/run.json", "visible_lip_sync": "not_assessed",
            "human_review": "pending_human_review", "export_authority": "none"}
        selection.gate.attempt.consume()
    except (Exception, KeyboardInterrupt):
        record = None
    if record is None:
        raise S2VError("S2V run preflight rejected or execution intent already consumed")
    failed = False
    try:
        directory = Path(output_dir)
        with PrivateRunDirectory(directory) as store:
            try:
                store.checkpoint(canonical(record))
                _admit(selection, verification, driver)
                endpoint = driver.endpoint()
                transport = driver.execution_transport()
                _admit(selection, verification, driver)
                remaining = min(deadline - clock(), driver.budget.remaining_seconds(driver._key) - 120)
                if remaining <= 0:
                    raise ValueError
                manifest = execute(selection.plan(), selection, verification, directory / "shot",
                    allow_live=True, endpoint=endpoint,
                    transport=transport, timeout_seconds=remaining, clock=clock,
                    result_timeout_seconds=min(30, remaining / 2))
                # Preserve paid output identity before fallible technical inspection.
                record.update(outcome="generated", manifest=manifest.to_dict())
                store.checkpoint(canonical(record))
                _admit(selection, verification, driver)
                if driver.budget.remaining_seconds(driver._key) <= 15:
                    raise ValueError
                measured = technical_result(selection, manifest, directory / "shot", inspect_media)
                _admit(selection, verification, driver)
                record.update(outcome="pending_human_review", technical_result=measured)
                store.checkpoint(canonical(record))
            except (Exception, KeyboardInterrupt):
                failed = True
                record["outcome"] = "incomplete"
                try:
                    store.checkpoint(canonical(record))
                except (Exception, KeyboardInterrupt):
                    pass
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed:
        raise S2VError("S2V run incomplete; inspect private run/shot state before any new request")
    return record
