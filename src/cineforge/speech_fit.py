"""Deterministic pitch-preserving stretch of over-slot mono PCM16 speech to its slot.

Under/at-slot input is returned unchanged; over-slot output fits exactly. No I/O.
"""
from __future__ import annotations

import math
import struct
from array import array

from cineforge.audio_pcm import inspect_pcm16_wav
from cineforge.native_speech_pcm import pcm_wav


class SpeechFitError(ValueError):
    """Static diagnostics without input bytes or private metadata."""


def fit_pcm16_wav_to_slot(payload: bytes, *, ticks_per_second: int, duration_ticks: int) -> bytes:
    wave = inspect_pcm16_wav(payload)
    if wave.channels != 1:
        raise SpeechFitError("speech fit requires mono PCM16")
    if (type(ticks_per_second) is not int or ticks_per_second <= 0
            or type(duration_ticks) is not int or duration_ticks <= 0):
        raise SpeechFitError("invalid speech slot timing")
    if duration_ticks * wave.sample_rate % ticks_per_second:
        raise SpeechFitError("speech slot requires an exact frame count")
    slot_frames = duration_ticks * wave.sample_rate // ticks_per_second
    if slot_frames <= 0:
        raise SpeechFitError("speech slot requires positive frames")
    if wave.frames <= slot_frames:
        return payload
    if wave.frames < 64:
        raise SpeechFitError("speech fit requires at least 64 source frames")
    samples: list[int] = list(struct.unpack(f"<{wave.frames}h", wave.pcm))
    window = 1024
    while window > 32 and window * 2 > wave.frames:
        window //= 2
    hop = window // 4
    radius = window // 4
    overlap = window - hop
    corr_len = min(overlap, 384)
    win = [0.5 - 0.5 * math.cos(2 * math.pi * k / window) for k in range(window)]
    out = array("d", [0.0]) * slot_frames
    weight = array("d", [0.0]) * slot_frames
    prev_start: int | None = None
    y = 0
    while y < slot_frames:
        nominal = y * wave.frames // slot_frames
        if prev_start is None:
            start = min(nominal, wave.frames - window)
        else:
            reference = samples[prev_start + hop:prev_start + hop + corr_len]
            best_score: int | None = None
            start = 0
            for delta in range(-radius, radius + 1):
                candidate = max(0, min(wave.frames - window, nominal + delta))
                score = sum(a * b for a, b in zip(
                    samples[candidate:candidate + corr_len], reference, strict=True))
                if best_score is None or score > best_score:
                    start, best_score = candidate, score
        for k in range(window):
            pos = y + k
            if pos < slot_frames:
                out[pos] += win[k] * samples[start + k]
                weight[pos] += win[k]
        prev_start = start
        y += hop
    pcm = bytearray(slot_frames * 2)
    for i in range(slot_frames):
        value = out[i] / weight[i] if weight[i] > 1e-8 else 0.0
        struct.pack_into("<h", pcm, i * 2, max(-32768, min(32767, round(value))))
    return pcm_wav(bytes(pcm), wave.sample_rate)
