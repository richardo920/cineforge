"""Bounded lossless review mux: VP9 picture plus measured PCM, no export authority."""
from __future__ import annotations

import hashlib
import math
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from cineforge import s2v_profile as profile
from cineforge.a20_inputs import A20Error
from cineforge.audio_analysis import FFmpegTool, ToolIdentity
from cineforge.audio_mix import MixedArtifact
from cineforge.audio_process import run_audio_process
from cineforge.private_store import PrivateRunDirectory, read_private_input
from cineforge.workflows import strict_object


def mux_delivery(video: Path, mix: MixedArtifact, store: PrivateRunDirectory, *,
                 ffmpeg: Path, ffprobe: Path, expected_tool: ToolIdentity,
                 deadline: float) -> dict[str, Any]:
    """Copy streams without trimming/re-timing; inspect both decoded output streams."""
    def remaining() -> float:
        value = min(30, deadline - time.monotonic())
        if value <= 0:
            raise A20Error("A20_DELIVERY_CAP")
        return value

    tool = FFmpegTool(ffmpeg)
    if (expected_tool.kind != "native" or expected_tool.version != "9.0.2"
            or tool.identify(timeout_seconds=remaining(), max_output_bytes=65536) != expected_tool
            or mix.report.delivery_status != "native_measured_pcm"):
        raise A20Error("A20_DELIVERY_TOOL")
    source = read_private_input(video, max_bytes=32 * 1024 * 1024)
    with tempfile.TemporaryDirectory(prefix="cineforge-a20-mux-") as temporary:
        root = Path(temporary).resolve()
        picture, soundtrack, output = root / "picture.webm", root / "sound.wav", root / "clip.mkv"
        picture.write_bytes(source)
        soundtrack.write_bytes(mix.payload)
        run_audio_process((str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
            "-protocol_whitelist", "file,pipe", "-i", str(picture),
            "-protocol_whitelist", "file,pipe", "-i", str(soundtrack),
            "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1",
            "-c:v", "copy", "-c:a", "pcm_s16le", "-f", "matroska", str(output)),
            payload=b"", timeout_seconds=remaining(), max_output_bytes=65536)
        facts = strict_object(run_audio_process((str(ffprobe), "-v", "error",
            "-max_alloc", "67108864", "-protocol_whitelist", "file,pipe",
            "-format_whitelist", "matroska,webm", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames,sample_rate,channels:format=duration",
            "-of", "json", str(output)), payload=b"", timeout_seconds=remaining(),
            max_output_bytes=65536))
        streams = facts["streams"]
        if not isinstance(streams, list) or len(streams) != 2:
            raise A20Error("A20_DELIVERY_STREAMS")
        v, a = streams
        duration = float(facts["format"]["duration"])
        if (v.get("codec_type") != "video" or v.get("codec_name") != "vp9"
                or v.get("width") != profile.WIDTH or v.get("height") != profile.HEIGHT
                or v.get("pix_fmt") != "yuv420p" or v.get("nb_read_frames") != "77"
                or Fraction(v.get("avg_frame_rate", "0")) != profile.FPS
                or a.get("codec_type") != "audio" or a.get("codec_name") != "pcm_s16le"
                or a.get("sample_rate") != str(mix.report.settings.sample_rate)
                or a.get("channels") != mix.report.settings.channels
                or not math.isfinite(duration) or abs(duration - profile.DURATION) > 1 / profile.FPS):
            raise A20Error("A20_DELIVERY_TECHNICAL")
        payload = read_private_input(output, max_bytes=64 * 1024 * 1024)
    remaining()
    if tool.identify(timeout_seconds=remaining(), max_output_bytes=65536) != expected_tool:
        raise A20Error("A20_DELIVERY_TOOL_CHANGED")
    return {"clip": store.put_payload(payload, "mkv"),
            "video_sha256": hashlib.sha256(source).hexdigest(),
            "mix": mix.pin.to_dict(), "mix_report": mix.report.to_dict(),
            "technical": facts, "human_review": "pending_human_review", "export_authority": "none"}
