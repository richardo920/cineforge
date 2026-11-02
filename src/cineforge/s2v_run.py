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
from cineforge.comfy_transport import request_deadline, require_time
from cineforge.contracts import RunManifest
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.s2v import S2VError, S2VSelection
from cineforge.s2v_service import S2VServiceContext
from cineforge.ssh_worker import SSHWorker, run_process
from cineforge.worker_bootstrap import validate_s2v_runtime
from cineforge.workflows import canonical, strict_object


def inspect_s2v_media(path: Path, *, absolute_deadline: float | None = None,
                      clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Fixed local Matroska/VP9 probe; no network protocols or arbitrary media URL."""
    candidate = None
    deadline = 0.0
    try:
        deadline = request_deadline(10, absolute_deadline, clock)
        binary = shutil.which("ffprobe")
        if binary is None:
            raise ValueError
        def check() -> None:
            require_time(deadline, clock)
        data = strict_object(run_process([binary, "-v", "error", "-max_alloc", "67108864",
            "-protocol_whitelist", "file,pipe", "-format_whitelist", "matroska,webm",
            "-f", "matroska", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames:format=duration",
            "-of", "json", str(path)], timeout=require_time(deadline, clock), max_output=65536,
            check=check, absolute_deadline=deadline))
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
        candidate = {"media_type": "video/webm", "codec": "vp9", "pixel_format": "yuv420p",
            "width": profile.WIDTH, "height": profile.HEIGHT, "decoded_frames": profile.FRAMES,
            "fps": float(profile.FPS), "duration_seconds": duration}
    except (Exception, KeyboardInterrupt):
        candidate = None
    # run_process has completed all owned-process cleanup before this checkout.
    # Never retain a candidate as success after any caught finalization failure.
    failed = False
    try:
        require_time(deadline, clock)
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or candidate is None:
        raise S2VError("bounded local S2V video inspection unavailable or rejected")
    return candidate


def technical_result(selection: S2VSelection, manifest: RunManifest, directory: Path,
                     inspect_media: Callable[[Path], dict[str, Any]] = inspect_s2v_media,
                     *, absolute_deadline: float | None = None,
                     clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Actual decoded output plus paired speech timing, not semantic or visible-sync review."""
    result = None
    try:
        def check() -> None:
            if absolute_deadline is not None:
                require_time(absolute_deadline, clock)
        check()
        selection.gate.preflight()
        check()
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
        check()
        media = (inspect_s2v_media(path, absolute_deadline=absolute_deadline, clock=clock)
                 if inspect_media is inspect_s2v_media else inspect_media(path))
        check()
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
        check()
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
        result = None
    failed = False
    try:
        if absolute_deadline is not None:
            require_time(absolute_deadline, clock)
    except (Exception, KeyboardInterrupt):
        failed = True
    if failed or result is None:
        raise S2VError("actual S2V video or paired speech timing contract rejected")
    return result


def _window(driver: SSHWorker, deadline: float, clock: Callable[[], float]) -> None:
    require_time(deadline, clock)
    remaining = driver.budget.remaining_seconds(driver._key)
    if type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining <= 120:
        raise S2VError("S2V attempt reached its owned lease cleanup reserve")
    require_time(deadline, clock)


def _admit(selection: S2VSelection, verification: WorkerVerification, driver: SSHWorker,
           deadline: float, clock: Callable[[], float]) -> None:
    _window(driver, deadline, clock)
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
    driver.verify_comfy_current(absolute_deadline=deadline)
    _window(driver, deadline, clock)


def run_s2v(selection: S2VSelection, verification: WorkerVerification, driver: SSHWorker,
            output_dir: str | Path, *, allow_live: bool = False,
            execute: Callable[..., RunManifest] = execute_comfy,
            inspect_media: Callable[[Path], dict[str, Any]] = inspect_s2v_media,
            absolute_deadline: float | None = None,
            clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """No retries. Caller keeps outer budget/guard/enforcer and guaranteed lease cleanup.

    Requires already checked paired staging; no private workflow reaches an unproven
    forward. Existing execute_comfy owns atomic cancellation and bounded result reserve.
    Any attempted run consumes this nonserialized intent, including lost acknowledgements.
    The durable journal retains generated evidence, never review readiness. Only the
    returned copy is finalized after all owned cleanup and fresh current admission.
    absolute_deadline is a trusted cap on this host's monotonic clock; it only
    shortens the phase/lease windows and is never transferred to another host.
    """
    record = None
    try:
        if allow_live is not True:
            raise ValueError
        # Capture the controller cap before any foreign proof/admission code.
        deadline = request_deadline(600, absolute_deadline, clock)
        sampled = clock()
        remaining = driver.budget.remaining_seconds(driver._key)
        if (type(remaining) not in (int, float) or not math.isfinite(remaining)
                or remaining <= 150):
            raise ValueError
        deadline = min(deadline, sampled + remaining - 120)
        _admit(selection, verification, driver, deadline, clock)
        trace = selection.gate.preflight()
        record = {"schema_version": 1, "mode": "m5-s2v-one-shot", "outcome": "prepared",
            "subject_sha256": selection.approved_subject_sha256,
            "scope_sha256": selection.gate.service.approval.subject_sha256,
            "lineage": trace, "worker_verification": verification.record,
            "worker_source_sha256": hashlib.sha256(driver._source).hexdigest(),
            "controller_source_sha256": hashlib.sha256(read_private_input(
                Path(__file__), max_bytes=128 * 1024)).hexdigest(),
            "runner_revision": profile.RUNNER_REVISION,
            "controller_deadline": deadline,
            "shot_record": "shot/run.json", "visible_lip_sync": "not_assessed",
            "human_review": "unavailable", "export_authority": "none"}
        _window(driver, deadline, clock)
        selection.gate.attempt.consume()
    except (Exception, KeyboardInterrupt):
        record = None
    if record is None:
        raise S2VError("S2V run preflight rejected or execution intent already consumed")
    failed = False
    def checkpoint(store: PrivateRunDirectory) -> None:
        _window(driver, deadline, clock)
        store.checkpoint(canonical(record))
        _window(driver, deadline, clock)
    try:
        directory = Path(output_dir)
        with PrivateRunDirectory(directory) as store:
            try:
                checkpoint(store)
                _admit(selection, verification, driver, deadline, clock)
                endpoint = driver.endpoint(absolute_deadline=deadline)
                transport = driver.execution_transport(absolute_deadline=deadline)
                _admit(selection, verification, driver, deadline, clock)
                remaining = min(deadline - clock(), driver.budget.remaining_seconds(driver._key) - 120)
                if remaining <= 0:
                    raise ValueError
                manifest = execute(selection.plan(), selection, verification, directory / "shot",
                    allow_live=True, endpoint=endpoint,
                    transport=transport, timeout_seconds=remaining, clock=clock,
                    result_timeout_seconds=min(30, remaining / 2), absolute_deadline=deadline)
                # Preserve paid output identity before fallible technical inspection.
                record.update(outcome="generated", manifest=manifest.to_dict())
                checkpoint(store)
                _admit(selection, verification, driver, deadline, clock)
                measured = technical_result(selection, manifest, directory / "shot", inspect_media,
                                            absolute_deadline=deadline, clock=clock)
                _admit(selection, verification, driver, deadline, clock)
                # A durable candidate is historical evidence, not a finalized run.
                # In particular, closing this directory can still fail or be late.
                record.update(outcome="generated", technical_candidate=measured,
                              human_review="unavailable")
                checkpoint(store)
            except (Exception, KeyboardInterrupt):
                failed = True
                record.update(outcome="incomplete", human_review="unavailable")
                record.pop("technical_result", None)
                try:
                    store.checkpoint(canonical(record))
                except (Exception, KeyboardInterrupt):
                    pass
    except (Exception, KeyboardInterrupt):
        failed = True
    finalized = None
    if not failed:
        try:
            # This is deliberately outside the storage context and every owned
            # finalizer. Never repair/reopen a closed journal to grant readiness.
            _admit(selection, verification, driver, deadline, clock)
            finalized = strict_object(canonical(record))
            candidate = finalized.pop("technical_candidate")
            finalized.update(outcome="pending_human_review", technical_result=candidate,
                             human_review="pending_human_review")
            finalized = strict_object(canonical(finalized))
            # Last observed checkout follows callbacks, validation and copying.
            _window(driver, deadline, clock)
        except (Exception, KeyboardInterrupt):
            finalized = None
    if failed or finalized is None:
        raise S2VError("S2V run incomplete; inspect private run/shot state before any new request")
    return finalized
