"""Pure bounded PCM16 WAV framing shared by audio consumers; no tool or file I/O."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


class PCMError(ValueError):
    """Static diagnostics without input bytes or private metadata."""


@dataclass(frozen=True, slots=True)
class PCM16Wave:
    pcm: bytes = field(repr=False)
    sample_rate: int
    channels: int

    @property
    def frames(self) -> int:
        return len(self.pcm) // (2 * self.channels)


def inspect_pcm16_wav(payload: bytes, *, max_input_bytes: int = 16 * 1024 * 1024,
                      max_duration_seconds: int = 60) -> PCM16Wave:
    """RIFF/fmt16/data only; mono/stereo, 16/24/44.1/48kHz, PCM little-endian16.

    The returned framing is not a rights, generation or content-identity assertion.
    """
    if (type(max_input_bytes) is not int or not 44 <= max_input_bytes <= 16 * 1024 * 1024
            or type(max_duration_seconds) is not int or not 1 <= max_duration_seconds <= 60):
        raise PCMError("invalid PCM inspection bounds")
    if (not isinstance(payload, bytes) or not 44 <= len(payload) <= max_input_bytes
            or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE"
            or int.from_bytes(payload[4:8], "little") != len(payload) - 8):
        raise PCMError("unsupported or malformed PCM WAV")
    offset = 12
    chunks: list[bytes] = []
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise PCMError("truncated PCM WAV")
        size = int.from_bytes(payload[offset + 4:offset + 8], "little")
        end = offset + 8 + size
        if end > len(payload):
            raise PCMError("truncated PCM WAV")
        if (len(chunks) >= 2 or payload[offset:offset + 4] != (b"fmt " if not chunks else b"data")
                or not chunks and size != 16):
            raise PCMError("unsupported PCM WAV chunks")
        chunks.append(payload[offset + 8:end])
        offset = end + (size % 2)
    if offset != len(payload) or len(chunks) != 2:
        raise PCMError("unsupported PCM WAV chunks")
    encoding, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", chunks[0])
    pcm = chunks[1]
    if (encoding != 1 or channels not in (1, 2) or rate not in (16000, 24000, 44100, 48000)
            or bits != 16 or align != channels * 2 or byte_rate != rate * align
            or not pcm or len(pcm) % align or len(pcm) // align > max_duration_seconds * rate):
        raise PCMError("unsupported PCM WAV format or duration")
    return PCM16Wave(pcm, rate, channels)
