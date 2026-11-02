"""Public recipe and synthetic framing only; no original actor, models or runtime."""
import hashlib
import struct
import traceback
import unittest
import zlib
from dataclasses import replace

from test_speech import wav_bytes

from cineforge import s2v_profile as profile
from cineforge.cloud_budget import BudgetApproval, CloudBudgetError
from cineforge.comfy_policy import CORE_CLASSES
from cineforge.devices import Precision
from cineforge.paired_media import PairedMediaError, image_metadata, speech_metadata
from cineforge.rights import EvidenceRef
from cineforge.s2v_service import S2VBudgetApproval, S2VServiceContext


def png(width=1651, height=953, *, color=2, compressed=True, ancillary=b""):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    raw = (b"\0" + b"\x20\x50\x80" * width) * height
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height,
            8, color, 0, 0, 0)) + (chunk(b"tEXt", ancillary) if ancillary else b"")
            + chunk(b"IDAT", zlib.compress(raw, 6 if compressed else 0)) + chunk(b"IEND", b""))


class S2VRecipeTests(unittest.TestCase):
    def test_closed_candidate_four_assets_and_storage_compute_distinction(self):
        self.assertEqual(hashlib.sha256(profile._CANDIDATE.encode()).hexdigest(),
                         profile.CANDIDATE_SHA256)
        graph = profile.graph("a" * 64, "b" * 64, "en")
        self.assertEqual(len(graph), 14)
        self.assertEqual(graph["162"]["inputs"]["audio"], "speech-" + "b" * 64 + ".wav")
        self.assertEqual(graph["160"]["inputs"]["image"], "actor-" + "a" * 64 + ".png")
        self.assertEqual(graph["153"]["inputs"]["length"], 77)
        self.assertEqual(graph["168"]["inputs"]["fps"], 16)
        self.assertEqual(profile.model().required_precision, Precision.BF16)
        self.assertIn("fp8-scaled", profile.STORAGE_PRECISION)
        self.assertEqual(len(profile.model_files()), 4)
        self.assertEqual(sum(item[1] for item in profile.FILES), 24015552011)
        self.assertNotIn("LoadAudio", CORE_CLASSES)
        self.assertNotIn("AudioEncoderLoader", CORE_CLASSES)
        self.assertNotIn("workflow", profile.service_declarations())
        self.assertNotIn("approval", profile.service_declarations())
        self.assertEqual(profile.graph("a" * 64, "b" * 64, "zh")["152"]["inputs"]["seed"], 5202)

    def test_service_context_exists_before_any_speech_or_graph_without_runtime_claim(self):
        receipt = EvidenceRef("private://synthetic/operator-scope", "d" * 64)
        approval = S2VBudgetApproval.for_scope("/private/tmp/synthetic-m5-budget", "EU-RO-1", receipt)
        service = S2VServiceContext(approval)
        service.preflight()
        self.assertEqual(service.worker_profile,
                         {"id": profile.PROFILE_ID, "scope_sha256": approval.subject_sha256})
        self.assertEqual(approval.disk_gb, 150)
        self.assertFalse(hasattr(service, "workflow"))
        self.assertFalse(hasattr(service, "speech"))
        self.assertFalse(hasattr(service, "live_ready"))
        self.assertEqual(service.model.required_precision, Precision.BF16)
        with self.assertRaises(CloudBudgetError):
            BudgetApproval(approval.subject_sha256, approval.model_workflow_sha256,
                approval.model, approval.image, approval.gpu_id, approval.region, "12.8",
                1, 1, 1, 1800, 1.25, 80)
        for changes in (dict(aggregate_usd=True), dict(aggregate_usd=6), dict(max_attempts=2),
                        dict(disk_gb=80), dict(model_workflow_sha256="a" * 64), dict(image="mutable"),
                        dict(journal_directory="/private/tmp/../CANARY")):
            with self.subTest(changes=list(changes)), self.assertRaises(CloudBudgetError):
                replace(approval, **changes)
        with self.assertRaises(CloudBudgetError) as caught:
            S2VBudgetApproval.for_scope("CANARY", "EU-RO-1", receipt, aggregate_usd=float("nan"))
        self.assertNotIn("CANARY", "".join(traceback.format_exception(caught.exception)))
        self.assertIsNone(caught.exception.__context__)

    def test_original_rgb_png_actual_dimensions_and_large_bounded_input(self):
        payload = png(compressed=False, ancillary=b"synthetic CANARY metadata")
        self.assertGreater(len(payload), 2 * 1024 * 1024)
        data = image_metadata(payload)
        self.assertEqual((data["width"], data["height"], data["channels"]), (1651, 953, 3))
        self.assertEqual(data["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(data["size_bytes"], len(payload))
        self.assertNotIn("CANARY", repr(data))
        for changed in (png(color=6), payload + b"CANARY", payload[:100],
                        b"X" * (8 * 1024 * 1024 + 1)):
            with self.assertRaises(PairedMediaError) as caught:
                image_metadata(changed)
            self.assertIsNone(caught.exception.__context__)
            self.assertNotIn("CANARY", str(caught.exception))

    def test_mono_speech_frame_limit_exact_and_no_conversion(self):
        for rate in (16000, 24000, 44100, 48000):
            # Largest whole-frame length not exceeding the exact77/16s bound.
            frames = rate * 77 // 16
            payload = wav_bytes(frames=frames, rate=rate)
            data = speech_metadata(payload)
            self.assertEqual((data["sample_rate"], data["frames"], data["channels"]),
                             (rate, frames, 1))
            self.assertEqual(data["sha256"], hashlib.sha256(payload).hexdigest())
            with self.assertRaises(PairedMediaError):
                speech_metadata(wav_bytes(frames=frames + 1, rate=rate))
        with self.assertRaises(PairedMediaError):
            speech_metadata(wav_bytes(channels=2))


if __name__ == "__main__":
    unittest.main()
