"""Explicit unit-range float to canonical PCM16 quantization; no model or tool imports."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from cineforge.native_speech_integrity import NativeSpeechError


@dataclass(frozen=True, slots=True)
class Conversion:
    frames: int
    sample_rate: int
    float_peak: float
    positive_endpoint_saturations: int


def pcm_wav(pcm: bytes, sample_rate: int) -> bytes:
    if (type(pcm) is not bytes or not pcm or len(pcm) % 2 or len(pcm) > 16 * 1024**2 - 44
            or type(sample_rate) is not int or sample_rate not in (16000, 24000, 44100, 48000)
            or len(pcm) // 2 > sample_rate * 60):
        raise NativeSpeechError("invalid native speech PCM framing")
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def float_pcm(samples: list[float], sample_rate: int) -> tuple[bytes, Conversion]:
    """Reject outside [-1,1]; round x*32768 ties-even, saturating +endpoint only.

    No gain, dither, resampling, padding, trimming or downmixing. Quantization does
    not prevent clipping/true-peak overload; those require independent analysis.
    """
    if (type(samples) is not list or type(sample_rate) is not int
            or sample_rate not in (16000, 24000, 44100, 48000)
            or not 1 <= len(samples) <= sample_rate * 60):
        raise NativeSpeechError("unsupported native speech waveform")
    pcm = bytearray(len(samples) * 2)
    peak = 0.0
    saturation = 0
    for index, value in enumerate(samples):
        if type(value) is not float or not math.isfinite(value) or not -1 <= value <= 1:
            raise NativeSpeechError("native speech waveform exceeds finite unit range")
        peak = max(peak, abs(value))
        rounded = round(value * 32768)
        saturation += rounded > 32767
        struct.pack_into("<h", pcm, index * 2, max(-32768, min(32767, rounded)))
    return pcm_wav(bytes(pcm), sample_rate), Conversion(len(samples), sample_rate, peak, saturation)
