"""Linux stat device identities are protocol data, not controller-native dev_t."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import test_native_driver as driver_fixtures

from cineforge import native_driver as driver


class LinuxDeviceCodecTests(unittest.TestCase):
    def test_fixed_linux_vectors_include_high_major_and_minor_bits(self):
        # Independently captured Linux CPython os.makedev/os.major/os.minor values.
        vectors = (
            (0, (0, 0)), (2049, (8, 1)), (64768, (253, 0)),
            (4293985279, (259, 1048575)), (1048575, (4095, 255)),
            (17592187092992, (4096, 256)), (4486076160802815, (1048575, 16777215)),
            (9223363240761753599, (2147483647, 2147483647)),
            (2**64-1, (2**32-1, 2**32-1)),
        )
        with patch.object(driver.os, "major", side_effect=AssertionError("controller ABI used")), \
                patch.object(driver.os, "minor", side_effect=AssertionError("controller ABI used")):
            for packed, expected in vectors:
                with self.subTest(packed=packed):
                    self.assertEqual(driver.linux_device_parts(packed), expected)

    def test_strict_uint64_boundary_and_types(self):
        for value in (-1, 2**64, 2**128, True, False, 0.0, 2049.0, "2049", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                driver.linux_device_parts(value)

    def test_linux_receipt_validates_with_incompatible_host_codec_but_mismatches_refuse(self):
        fixture = driver_fixtures.DriverMountTests("runTest")
        fixture.setUp()
        try:
            binary = fixture.measure()
            witness = binary["driver_mount"]
            self.assertEqual(witness["identity"][0], 2049)
            self.assertEqual(witness["mount"][2], "8:1")
            def validate(value):
                driver.validate_driver_receipt(value, path=binary["path"], size_bytes=binary["size_bytes"],
                    sha256=binary["sha256"], binding_sha256=witness["binding_sha256"])
            for major, minor in ((lambda dev: 0, lambda dev: dev),
                                 (lambda dev: (_ for _ in ()).throw(OverflowError()),
                                  lambda dev: (_ for _ in ()).throw(OverflowError()))):
                with patch.object(driver.os, "major", side_effect=major), \
                        patch.object(driver.os, "minor", side_effect=minor):
                    validate(witness)
                    for field, value in (("mount", "0:2049"), ("device", 2050), ("device", -1),
                                         ("device", 2**64), ("device", True)):
                        invalid = copy.deepcopy(witness)
                        if field == "mount":
                            invalid["mount"][2] = value
                        else:
                            invalid["identity"][0] = value
                        with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                            validate(invalid)
        finally:
            fixture.doCleanups()
