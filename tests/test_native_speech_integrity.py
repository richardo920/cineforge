"""Offline admission checks for torch-observed hardware and driver reservations."""

import unittest

from cineforge.native_speech_integrity import NativeHardware, NativeSpeechError


class NativeHardwareMatchesTests(unittest.TestCase):
    def setUp(self):
        self.hardware = NativeHardware("yxqf2tbemhpmmt", "NVIDIA L40S", 46068, (8, 9))
        self.actual = {
            "torch_version": "2.8.0+cu128",
            "torchaudio_version": "2.8.0+cu128",
            "cuda_version": "12.8",
            "gpu_name": "NVIDIA L40S",
            "memory_bytes": 46068 * 1024**2,
            "capability": [8, 9],
            "operator_worker_id": "yxqf2tbemhpmmt",
            "bf16_supported": True,
            "attention": "sdpa-math",
        }

    def test_exact_match_passes(self):
        self.hardware.require_matches(self.actual)

    def test_live_l40s_driver_reservation_passes(self):
        self.actual["memory_bytes"] = 47_667_740_672
        self.assertEqual(self.actual["memory_bytes"] // 1024**2, 45459)
        self.hardware.require_matches(self.actual)
        NativeHardware("yxqf2tbemhpmmt", "NVIDIA L40S", 45459, (8, 9)).require_matches(
            self.actual
        )

    def test_tolerance_boundary_passes(self):
        self.actual["memory_bytes"] = (46068 - 2048) * 1024**2
        self.hardware.require_matches(self.actual)

    def test_beyond_tolerance_fails(self):
        self.actual["memory_bytes"] = (46068 - 2049) * 1024**2
        with self.assertRaises(NativeSpeechError):
            self.hardware.require_matches(self.actual)

    def test_above_expected_fails(self):
        self.actual["memory_bytes"] = (46068 + 1) * 1024**2
        with self.assertRaises(NativeSpeechError):
            self.hardware.require_matches(self.actual)

    def test_other_fields_remain_exact_with_tolerated_memory(self):
        self.actual["memory_bytes"] = 47_667_740_672
        for field, value in (
            ("gpu_name", "NVIDIA L4"),
            ("capability", [8, 6]),
            ("capability", (8, 9)),
            ("capability", [8, 9.0]),
            ("torch_version", "2.8.0"),
            ("torchaudio_version", "2.8.0"),
            ("cuda_version", "12.7"),
            ("bf16_supported", False),
            ("bf16_supported", 1),
            ("attention", "flash-attention"),
            ("operator_worker_id", "another-worker"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(NativeSpeechError):
                self.hardware.require_matches({**self.actual, field: value})

    def test_key_set_remains_exact(self):
        for field in self.actual:
            with self.subTest(missing=field), self.assertRaises(NativeSpeechError):
                self.hardware.require_matches({k: v for k, v in self.actual.items() if k != field})
        with self.assertRaises(NativeSpeechError):
            self.hardware.require_matches({**self.actual, "unexpected": True})

    def test_memory_type_remains_integer(self):
        for value in (47_667_740_672.0, "47667740672", True, None):
            with self.subTest(value=value), self.assertRaises(NativeSpeechError):
                self.hardware.require_matches({**self.actual, "memory_bytes": value})


if __name__ == "__main__":
    unittest.main()
