"""Bounded long-form review delivery; stream-copy picture, linear PCM mix, no export authority."""
from __future__ import annotations

import hashlib
import math
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from cineforge import s2v_profile as profile
from cineforge.a20_inputs import A20Error
from cineforge.audio import CueSheet, RevisionPin
from cineforge.audio_analysis import FFmpegTool, ToolIdentity
from cineforge.audio_mix import INTERNAL_REVIEW_STEREO, MixSettings, mix_scene
from cineforge.audio_pcm import MAX_DURATION_SECONDS, MAX_PCM_BYTES, inspect_pcm16_wav
from cineforge.audio_process import run_audio_process
from cineforge.characters import CharacterRegistry
from cineforge.contracts import ProjectPlan
from cineforge.json_codec import canonical_json
from cineforge.private_store import VIDEO_MAX_BYTES, PrivateRunDirectory, read_private_input
from cineforge.rights import RightsLedger
from cineforge.s2v_run import inspect_s2v_media
from cineforge.speech import SpeechArtifact
from cineforge.stem_import import ImportedStemArtifact
from cineforge.workflows import strict_object

# Metrics use 64x40 grayscale thumbnails in [0,255]. MAD > 80 is informational;
# temporal mean squared difference <= .01 marks a near-static/frozen segment.
HARD_CUT_MAD = 80.0
FROZEN_VARIANCE = .01
_SEGMENT_BYTES = 32 * 1024 * 1024
_PIECE_BYTES = VIDEO_MAX_BYTES
_RAW_BYTES = profile.WIDTH * profile.HEIGHT * 3 // 2 * profile.FRAMES
FrameTransform = Callable[[bytes, int], bytes]


def _remaining(deadline: float) -> float:
    value = min(30, deadline - time.monotonic())
    if not math.isfinite(value) or value <= 0:
        raise A20Error("LONG_DELIVERY_CAP")
    return value


def _run(command: Sequence[str], deadline: float) -> bytes:
    return run_audio_process(tuple(command), payload=b"", timeout_seconds=_remaining(deadline),
                             max_output_bytes=65536)


def _encode(ffmpeg: Path, args: Sequence[str], deadline: float) -> None:
    _run((str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", *args), deadline)


def _probe(path: Path, ffprobe: Path, deadline: float, frames: int,
           audio: str | None = None, video: str = "vp9") -> dict[str, Any]:
    facts = strict_object(_run((str(ffprobe), "-v", "error", "-max_alloc", "67108864",
        "-protocol_whitelist", "file,pipe", "-count_frames", "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
        "-of", "json", str(path)), deadline))
    streams = facts["streams"]
    duration = float(facts["format"]["duration"])
    if not isinstance(streams, list) or len(streams) != (2 if audio else 1):
        raise A20Error("LONG_DELIVERY_STREAMS")
    v = streams[0]
    if (v.get("codec_type") != "video" or v.get("codec_name") != video
            or v.get("width") != profile.WIDTH or v.get("height") != profile.HEIGHT
            or v.get("pix_fmt") != "yuv420p" or v.get("nb_read_frames") != str(frames)
            or Fraction(v.get("avg_frame_rate", "0")) != profile.FPS
            or not math.isfinite(duration) or abs(duration - frames / profile.FPS) > 1 / profile.FPS
            or audio and (streams[1].get("codec_type") != "audio"
                          or streams[1].get("codec_name") != audio)):
        raise A20Error("LONG_DELIVERY_TECHNICAL")
    return facts


def _count(shots: Sequence[Path | bytes]) -> int:
    if not 1 <= len(shots) <= int(MAX_DURATION_SECONDS / profile.DURATION):
        raise A20Error("LONG_DELIVERY_SEGMENTS")
    return len(shots)


def _stitch(shots: Sequence[Path | bytes], root: Path, *, ffmpeg: Path, ffprobe: Path,
            deadline: float, transform: FrameTransform | None) -> tuple[Path, list[Path]]:
    count = _count(shots)
    segments = []
    for index, shot in enumerate(shots):
        payload = shot if isinstance(shot, bytes) else read_private_input(shot, max_bytes=_SEGMENT_BYTES)
        if not 0 < len(payload) <= _SEGMENT_BYTES:
            raise A20Error("LONG_DELIVERY_SEGMENT_BYTES")
        path = root / f"segment-{index}.webm"
        path.write_bytes(payload)
        inspect_s2v_media(path, absolute_deadline=deadline)
        if transform is not None:
            # The hook receives all 77 packed yuv420p frames at native resolution.
            raw = root / "frames.yuv"
            _encode(ffmpeg, ("-protocol_whitelist", "file,pipe", "-i", str(path),
                "-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", str(raw)), deadline)
            frames = read_private_input(raw, max_bytes=_RAW_BYTES)
            if len(frames) != _RAW_BYTES:
                raise A20Error("LONG_DELIVERY_TRANSFORM_FRAMES")
            changed = transform(frames, index)
            if not isinstance(changed, bytes) or len(changed) != _RAW_BYTES:
                raise A20Error("LONG_DELIVERY_TRANSFORM_FRAMES")
            raw.write_bytes(changed)
            path = root / f"transformed-{index}.webm"
            _encode(ffmpeg, ("-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "832x480",
                "-framerate", "16", "-i", str(raw), "-an", "-c:v", "libvpx-vp9", "-lossless", "1",
                "-threads", "2", "-speed", "8", str(path)), deadline)
            raw.unlink()
            inspect_s2v_media(path, absolute_deadline=deadline)
        segments.append(path)
    manifest = root / "concat.txt"
    manifest.write_text("ffconcat version 1.0\n" + "".join(
        f"file '{path.name}'\nduration {profile.DURATION}\n" for path in segments))
    output = root / "stitched.webm"
    _encode(ffmpeg, ("-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i",
        str(manifest), "-map", "0:v:0", "-map_metadata", "-1", "-c:v", "copy", str(output)), deadline)
    _probe(output, ffprobe, deadline, count * profile.FRAMES)
    return output, segments


def stitch_shots(segment_shots: Sequence[Path | bytes], *, ffmpeg: Path, ffprobe: Path,
                 deadline: float, transform: FrameTransform | None = None) -> bytes:
    """Return stream-copied VP9 WebM. Hook: packed 77-frame yuv420p bytes, zero-based index.

    None is the identity fast path. Supplied hooks are losslessly re-encoded and validated.
    Input validation is technical only; callers retain shot provenance/review responsibility.
    """
    with tempfile.TemporaryDirectory(prefix="cineforge-long-stitch-") as temporary:
        output, _ = _stitch(segment_shots, Path(temporary).resolve(), ffmpeg=ffmpeg,
                            ffprobe=ffprobe, deadline=deadline, transform=transform)
        return read_private_input(output, max_bytes=_PIECE_BYTES)


def _qc(segments: list[Path], root: Path, ffmpeg: Path, deadline: float) -> dict[str, Any]:
    boundaries: list[dict[str, Any]] = []
    reports = []
    previous = None
    size = 64 * 40
    for index, segment in enumerate(segments):
        raw = root / f"qc-{index}.gray"
        _encode(ffmpeg, ("-protocol_whitelist", "file,pipe", "-i", str(segment), "-map", "0:v:0",
            "-vf", "scale=64:40", "-pix_fmt", "gray", "-f", "rawvideo", str(raw)), deadline)
        data = read_private_input(raw, max_bytes=size * profile.FRAMES)
        if len(data) != size * profile.FRAMES:
            raise A20Error("LONG_DELIVERY_QC_FRAMES")
        first, last = data[:size], data[-size:]
        variance = sum((a - b) ** 2 for a, b in zip(data[size:], data[:-size], strict=True)) / (len(data) - size)
        reports.append({"segment_index": index, "temporal_variance": variance,
                        "flags": ["frozen_suspect"] if variance <= FROZEN_VARIANCE else []})
        if previous is not None:
            mad = sum(abs(a - b) for a, b in zip(previous, first, strict=True)) / size
            hist_a, hist_b = [0] * 16, [0] * 16
            for a, b in zip(previous, first, strict=True):
                hist_a[a // 16] += 1
                hist_b[b // 16] += 1
            boundaries.append({"left_segment": index - 1, "right_segment": index,
                "mean_absolute_pixel_difference": mad,
                "histogram_distance": sum(abs(a - b) for a, b in zip(hist_a, hist_b, strict=True)) / (2 * size),
                "flags": ["hard_cut_outlier"] if mad > HARD_CUT_MAD else []})
        previous = last
        raw.unlink()
    return {"boundaries": boundaries, "segments": reports,
            "thresholds": {"hard_cut_mad": HARD_CUT_MAD, "frozen_temporal_variance": FROZEN_VARIANCE},
            "metric": "64x40 grayscale; 16-bin normalized histogram total variation"}


def mux_long_delivery(segment_shots: Sequence[Path | bytes], sheet: CueSheet,
                      sources: tuple[SpeechArtifact | ImportedStemArtifact, ...],
                      settings: MixSettings, store: PrivateRunDirectory, *,
                      plan: ProjectPlan, registry: CharacterRegistry, ledger: RightsLedger,
                      current_picture: RevisionPin, current_script: RevisionPin,
                      output_id: str, output_revision: int, ffmpeg: Path, ffprobe: Path,
                      expected_tool: ToolIdentity, deadline: float,
                      transform: FrameTransform | None = None) -> dict[str, Any]:
    """Mix the full cue timeline, copy VP9/PCM master, derive H.264/AAC review and QC.

    Supply one dialogue cue per segment at i*77/16; shorter speech leaves silence.
    Music/ambience sources must span the piece; foley retains its cue timing.
    """
    count = _count(segment_shots)
    ticks = sheet.timebase.ticks_per_second
    duration = count * profile.DURATION
    if sheet.timebase.duration_ticks != duration * ticks:
        raise A20Error("LONG_DELIVERY_TIMELINE")
    dialogue = sorted((cue for cue in sheet.cues if cue.kind == "dialogue"),
                      key=lambda cue: cue.start_ticks)
    if len(dialogue) != count or any(cue.start_ticks != i * profile.DURATION * ticks
            or cue.duration_ticks != profile.DURATION * ticks for i, cue in enumerate(dialogue)):
        raise A20Error("LONG_DELIVERY_DIALOGUE")
    by_cue = {source.cue.cue_id: source for source in sources
              if isinstance(source, ImportedStemArtifact)}
    for cue in sheet.cues:
        if cue.kind in ("music", "ambience"):
            bed = by_cue.get(cue.cue_id)
            if (cue.start_ticks != 0 or cue.duration_ticks != sheet.timebase.duration_ticks
                    or bed is None or bed.frames != duration * bed.sample_rate):
                raise A20Error("LONG_DELIVERY_BED")
    tool = FFmpegTool(ffmpeg)
    if (expected_tool.kind != "native" or expected_tool.version != "9.0.2"
            or tool.identify(timeout_seconds=_remaining(deadline), max_output_bytes=65536) != expected_tool):
        raise A20Error("LONG_DELIVERY_TOOL")
    # mix_scene owns rights, exact sample placement, ducking, final linear gain and remeasurement.
    _remaining(deadline)
    bounded_settings = replace(settings, timeout_seconds=min(settings.timeout_seconds, deadline - time.monotonic()))
    mix = mix_scene(sheet, sources, bounded_settings, INTERNAL_REVIEW_STEREO, plan=plan,
        registry=registry, ledger=ledger, current_picture=current_picture, current_script=current_script,
        output_id=output_id, output_revision=output_revision, tool=tool, expected_tool=expected_tool)
    rms = []
    for cue in dialogue:
        source = next((source for source in sources if isinstance(source, SpeechArtifact)
            and source.request.utterance.input_id == cue.utterance_id), None)
        if source is None:
            raise A20Error("LONG_DELIVERY_DIALOGUE_SOURCE")
        wave = inspect_pcm16_wav(source.payload, max_input_bytes=MAX_PCM_BYTES,
                                max_duration_seconds=int(MAX_DURATION_SECONDS), long_form=True)
        energy = sum(int.from_bytes(wave.pcm[i:i + 2], "little", signed=True) ** 2
                     for i in range(0, len(wave.pcm), 2))
        rms.append(math.sqrt(energy / (len(wave.pcm) // 2)) / 32768)
    with tempfile.TemporaryDirectory(prefix="cineforge-long-delivery-") as temporary:
        root = Path(temporary).resolve()
        picture, segments = _stitch(segment_shots, root, ffmpeg=ffmpeg, ffprobe=ffprobe,
                                    deadline=deadline, transform=transform)
        sound, master, review = root / "mix.wav", root / "master.mkv", root / "review.mp4"
        sound.write_bytes(mix.payload)
        _encode(ffmpeg, ("-protocol_whitelist", "file,pipe", "-i", str(picture),
            "-protocol_whitelist", "file,pipe", "-i", str(sound), "-map", "0:v:0", "-map", "1:a:0",
            "-map_metadata", "-1", "-c:v", "copy", "-c:a", "pcm_s16le", str(master)), deadline)
        facts = _probe(master, ffprobe, deadline, count * profile.FRAMES, "pcm_s16le")
        if (facts["streams"][1].get("sample_rate") != str(settings.sample_rate)
                or facts["streams"][1].get("channels") != settings.channels):
            raise A20Error("LONG_DELIVERY_AUDIO")
        _encode(ffmpeg, ("-protocol_whitelist", "file,pipe", "-i", str(master), "-map", "0:v:0",
            "-map", "0:a:0", "-map_metadata", "-1", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "18", "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(review)), deadline)
        review_facts = _probe(review, ffprobe, deadline, count * profile.FRAMES, "aac", "h264")
        qc = _qc(segments, root, ffmpeg, deadline)
        qc.update(total_frames=count * profile.FRAMES, duration_seconds=duration,
                  integrated_lufs=mix.report.delivered.integrated_lufs.value,
                  dialogue_rms=rms, dialogue_rms_basis="source PCM normalized full-scale, before mix gain",
                  human_review="pending_human_review", export_authority="none")
        master_payload = read_private_input(master, max_bytes=_PIECE_BYTES)
        review_payload = read_private_input(review, max_bytes=_PIECE_BYTES)
    if tool.identify(timeout_seconds=_remaining(deadline), max_output_bytes=65536) != expected_tool:
        raise A20Error("LONG_DELIVERY_TOOL_CHANGED")
    return {"master": store.put_payload(master_payload, "mkv"),
            "review": store.put_payload(review_payload, "mp4"),
            "qc_path": store.put_payload(canonical_json(qc).encode(), "json"), "qc": qc,
            "master_sha256": hashlib.sha256(master_payload).hexdigest(),
            "mix": mix.pin.to_dict(), "mix_report": mix.report.to_dict(),
            "technical": facts, "review_technical": review_facts,
            "human_review": "pending_human_review", "export_authority": "none"}
