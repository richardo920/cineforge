"""Exercise real default media callbacks at the synthetic ffprobe JSON boundary."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_cloud_smoke as smoke_fixtures
import test_conditioning as conditioning_fixtures
from test_conditioning import Fixture

from cineforge.cloud_smoke import (
    CloudSmokeError,
    load_reference,
    normalize,
    probe_media,
    read_media,
)
from cineforge.conditioning import ConditioningError
from cineforge.conditioning_run import technical_result
from cineforge.contracts import RunManifest
from cineforge.workflows import canonical


def probe(m4=True):
    return {"streams": [{"codec_name": "vp9", "width": 1280 if m4 else 832,
        "height": 704 if m4 else 480, "pix_fmt": "yuv420p",
        "avg_frame_rate": "24/1" if m4 else "16/1", "nb_read_frames": "49" if m4 else "33"}],
        "format": {"duration": "2.042" if m4 else "2.063"}}


class DefaultM4MediaTests(Fixture):
    run_case = conditioning_fixtures.ManagedTests.run_case

    def test_default_managed_m4_reader_accepts_m4_and_still_leaves_human_review_pending(self):
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(probe())) as process:
            report = self.run_case(use_default_media=True)
        self.assertEqual(process.call_count, 3)
        self.assertTrue(all("-count_frames" in call.args[0] for call in process.call_args_list))
        self.assertEqual(report["outcome"], "pending_human_review")
        self.assertEqual(report["identity_review"], "pending_human_review")
        self.assertTrue(report["release_verified"])
        self.assertEqual(len(report["technical_results"]), 3)
        self.assertEqual(report["technical_results"][0]["media"]["decoded_frames"], 49)

    def test_m4_default_rejects_m3_metadata_after_owned_release(self):
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(probe(False))), \
                self.assertRaises(ConditioningError):
            self.run_case(use_default_media=True)
        report = json.loads((self.root / "run/run.json").read_bytes())
        self.assertEqual(report["outcome"], "incomplete")
        self.assertTrue(report["release_verified"])
        self.assertFalse(self.cloud.pods)
        self.assertEqual(len(report["shots"]), 3)

    def test_m4_technical_default_rejects_wrong_contract_and_malformed_measured_fields(self):
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(probe())):
            report = self.run_case(use_default_media=True)
        first = report["shots"][0]
        manifest = RunManifest.from_dict(first["manifest"])
        directory = self.root / "run" / first["shot_id"]
        cases = []
        for field, values in {
            "width": (832, True, 1280.0), "height": (480, 0), "codec_name": ("h264",),
            "pix_fmt": ("rgb24",), "nb_read_frames": ("48", "50", "0", "49.0", "N/A", 49, True),
            "avg_frame_rate": ("16/1", "24/0", "0/1", "NaN", "1e999999999", "9"*41, 24),
        }.items():
            for value in values:
                data = probe()
                data["streams"][0][field] = value
                cases.append(data)
        for value in ("nan", "inf", "-inf", "0", "1", True, None, [], "9"*41):
            data = probe()
            data["format"]["duration"] = value
            cases.append(data)
        cases += [{}, {"streams": [], "format": {}}, {"streams": [None], "format": {}},
                  {"streams": probe()["streams"]*2, "format": {"duration": "2.042"}}]
        for index, data in enumerate(cases):
            with self.subTest(case=index), \
                    patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                    patch("cineforge.cloud_smoke.run_process", return_value=canonical(data)), \
                    self.assertRaises((CloudSmokeError, ConditioningError)):
                technical_result(manifest, directory)  # Actual default, no inspect_media injection.


class DefaultM3MediaTests(unittest.TestCase):
    setUp = smoke_fixtures.ManagedSmokeTests.setUp

    def test_m3_default_probe_normalize_and_reference_stay_strict_in_both_directions(self):
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(probe(False))):
            self.assertEqual(probe_media(Path("fixture.webm"))["decoded_frames"], 33)
            result = normalize(self.manifest, self.selection, self.root / "baseline")
            self.assertEqual(result["contract"]["media"]["width"], 832)
            self.assertEqual(load_reference(self.root / "baseline/run.json", self.selection).normalized[
                "contract"], self.reference.normalized["contract"])
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(probe())):
            self.assertEqual(read_media(Path("fixture.webm"))["width"], 1280)
            for call in (
                lambda: probe_media(Path("fixture.webm")),
                lambda: normalize(self.manifest, self.selection, self.root / "baseline"),
                lambda: load_reference(self.root / "baseline/run.json", self.selection),
            ):
                with self.assertRaises(CloudSmokeError):
                    call()

    def test_shared_reader_rejects_nonfinite_malformed_or_unavailable_probe_without_raw_text(self):
        for raw in (b'{"streams":null}', b'{"format":NaN}', b'[]', b'not-json'):
            with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                    patch("cineforge.cloud_smoke.run_process", return_value=raw), \
                    self.assertRaises(CloudSmokeError):
                read_media(Path("fixture.webm"))
        with patch("cineforge.cloud_smoke.shutil.which", return_value=None), \
                patch("cineforge.cloud_smoke.run_process") as run, self.assertRaises(CloudSmokeError):
            read_media(Path("fixture.webm"))
        run.assert_not_called()
        bad = copy.deepcopy(probe(False))
        bad["streams"][0]["nb_read_frames"] = "33.0"
        with patch("cineforge.cloud_smoke.shutil.which", return_value="/fixture/ffprobe"), \
                patch("cineforge.cloud_smoke.run_process", return_value=canonical(bad)), \
                self.assertRaises(CloudSmokeError):
            probe_media(Path("fixture.webm"))
