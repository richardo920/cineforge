"""Exact-type digest syntax and caller-owned diagnostics, including isolated workers."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

from cineforge.json_codec import is_sha256_hex
from cineforge.native_driver import DRIVER_ALIASES, DRIVER_POLICY, validate_driver_receipt
from cineforge.native_remote_plan import nr_hash
from cineforge.s2v import S2VError, _sha
from cineforge.s2v_stage import _sha as stage_sha
from cineforge.worker_bundle import worker_source
from tests.stable_test_temp import stable_temporary_directory


class StringSubclass(str):
    pass


BAD_DIGESTS = ("A" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 63 + "\n",
               1, None, b"a" * 64, StringSubclass("a" * 64))


def receipt_arguments():
    alias, role = next(iter(DRIVER_ALIASES.items()))
    receipt = dict.fromkeys(("user_namespace", "mount_namespace", "uid_map", "mount",
                             "identity", "links", "ancestors"))
    receipt.update(policy=DRIVER_POLICY, alias=alias, role=role, binding_sha256="a" * 64)
    return dict(receipt=receipt, path=alias, size_bytes=1, sha256="b" * 64,
                binding_sha256="a" * 64)


class Sha256HexTests(unittest.TestCase):
    def test_plain_string_truth_table_including_zero_and_subclass(self):
        for value in ("0123456789abcdef" * 4, "0" * 64):
            self.assertIs(is_sha256_hex(value), True)
        for value in BAD_DIGESTS:
            with self.subTest(value=value):
                self.assertIs(is_sha256_hex(value), False)

    def test_sha_wrappers_preserve_exact_exception_types_and_messages(self):
        for check, kind, message in ((_sha, S2VError, "invalid S2V evidence digest"),
                                      (stage_sha, ValueError, ""),
                                      (nr_hash, ValueError, "invalid remote speech identity")):
            self.assertIsNone(check("a" * 64))
            for value in BAD_DIGESTS:
                with self.subTest(check=check, value=value), self.assertRaises(kind) as caught:
                    check(value)
                self.assertIs(type(caught.exception), kind)
                self.assertEqual(str(caught.exception), message)
        self.assertIsNone(_sha("0" * 64))
        self.assertIsNone(stage_sha("0" * 64))
        with self.assertRaisesRegex(ValueError, "^invalid remote speech identity$"):
            nr_hash("0" * 64)

    def test_both_driver_digest_checks_keep_diagnostic_and_zero_rejection(self):
        for field in ("binding_sha256", "sha256"):
            for value in (*BAD_DIGESTS, "0" * 64):
                arguments = receipt_arguments()
                arguments[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError) as caught:
                    validate_driver_receipt(**arguments)
                self.assertIs(type(caught.exception), ValueError)
                self.assertEqual(str(caught.exception), "invalid fixed NVIDIA receipt")

    def test_shared_validators_work_in_isolated_worker_bundle(self):
        source = worker_source().rsplit(b'\nif __name__ == "__main__":', 1)[0]
        probe = '''
results = []
for value in ("a"*64, "0"*64, "A"*64, None):
    results.append(is_sha256_hex(value))
for check in (_sha, nr_hash):
    try:
        check(None)
    except ValueError as error:
        results.append([type(error).__name__, str(error)])
for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
    try:
        _pair_json(raw)
    except ValueError:
        results.append("rejected JSON")
print(json.dumps(results))
'''
        with stable_temporary_directory(prefix="cf-sha256-bundle-") as temporary:
            script = Path(temporary) / "worker_probe.py"
            script.write_text(source.decode() + probe, encoding="utf-8")
            result = subprocess.run([sys.executable, "-I", str(script)],
                                    capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [True, True, False, False,
                         ["ValueError", ""], ["ValueError", "invalid remote speech identity"],
                         "rejected JSON", "rejected JSON"])
