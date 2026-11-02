"""Bounded PNG/mono PCM16 framing for private S2V pairs; no path/network/model I/O."""
from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Any

from cineforge.audio_pcm import inspect_pcm16_wav

PNG_MAX_BYTES = 8 * 1024 * 1024
SPEECH_MAX_BYTES = 1024 * 1024


class PairedMediaError(ValueError):
    """Static media diagnostics; never includes bytes or supplied metadata."""


def image_metadata(payload: bytes) -> dict[str, Any]:
    failed = False
    result = None
    try:
        if (not isinstance(payload, bytes) or not 0 < len(payload) <= PNG_MAX_BYTES
                or payload[:8] != b"\x89PNG\r\n\x1a\n"):
            raise ValueError
        offset, phase, width, height = 8, 0, 0, 0
        compressed = bytearray()
        ended = False
        while offset < len(payload):
            if offset + 12 > len(payload):
                raise ValueError
            length = int.from_bytes(payload[offset:offset + 4], "big")
            kind = payload[offset + 4:offset + 8]
            end = offset + 12 + length
            if (end > len(payload) or any(not (65 <= value <= 90 or 97 <= value <= 122)
                                         for value in kind)):
                raise ValueError
            value = payload[offset + 8:end - 4]
            if zlib.crc32(kind + value) & 0xffffffff != int.from_bytes(payload[end - 4:end], "big"):
                raise ValueError
            if kind == b"IHDR" and phase == 0:
                if len(value) != 13:
                    raise ValueError
                width, height, depth, color, compression, filtering, interlace = struct.unpack(
                    ">IIBBBBB", value)
                if (not 16 <= width <= 4096 or not 16 <= height <= 4096 or depth != 8
                        or color != 2 or compression or filtering or interlace):
                    raise ValueError
                phase = 1
            elif kind == b"IDAT" and phase in (1, 2):
                compressed.extend(value)
                phase = 2
            elif kind == b"IEND" and phase in (2, 3) and not length and end == len(payload):
                ended = True
            elif (phase != 0 and kind[0] & 32 and kind not in (b"acTL", b"fcTL", b"fdAT")):
                # Bounded ancillary metadata is ignored, never interpreted or fetched.
                phase = 3 if phase in (2, 3) else 1
            else:
                raise ValueError
            offset = end
        expected = (width * 3 + 1) * height
        decoder = zlib.decompressobj()
        raw = decoder.decompress(bytes(compressed), expected + 1)
        if (not ended or len(raw) != expected or not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail
                or any(raw[index] > 4 for index in range(0, expected, width * 3 + 1))):
            raise ValueError
        digest = hashlib.sha256(payload).hexdigest()
        result = {"sha256": digest, "size_bytes": len(payload), "basename": "actor-" + digest + ".png",
                  "width": width, "height": height, "channels": 3, "bit_depth": 8}
    except (ValueError, TypeError, OverflowError, struct.error, zlib.error):
        failed = True
    if failed or result is None:
        raise PairedMediaError("unsupported or malformed original actor PNG")
    return result


def speech_metadata(payload: bytes) -> dict[str, Any]:
    result = None
    failed = False
    try:
        wave = inspect_pcm16_wav(payload, max_input_bytes=SPEECH_MAX_BYTES,
                                max_duration_seconds=5)
        if wave.channels != 1 or wave.frames * 16 > 77 * wave.sample_rate:
            raise ValueError
        digest = hashlib.sha256(payload).hexdigest()
        result = {"sha256": digest, "size_bytes": len(payload), "basename": "speech-" + digest + ".wav",
                  "sample_rate": wave.sample_rate, "channels": 1, "frames": wave.frames,
                  "sample_format": "pcm_s16le"}
    except (ValueError, TypeError, OverflowError):
        failed = True
    if failed or result is None:
        raise PairedMediaError("unsupported canonical S2V speech WAV")
    return result
