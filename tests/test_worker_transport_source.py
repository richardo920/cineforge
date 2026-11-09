"""Real fixed bundle rendering/framing, without SSH, GPU or model imports."""
from __future__ import annotations

import ast
import base64
import hashlib
import shlex
import sys
import types
import unittest
import zlib

from cineforge.native_remote_package import native_package
from cineforge.worker_bundle import worker_source
from cineforge.worker_transport_source import SOURCE_LIMIT, worker_command_source


class FixedWorkerSourceTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = worker_source()
        cls.rendered = worker_command_source(cls.source)
        tree = ast.parse(cls.rendered)
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "_c")
        cls.encoded = ast.literal_eval(assignment.value.args[0])

    def execute(self, rendered):
        module = types.ModuleType("_fixed_worker_decode_test")
        sys.modules[module.__name__] = module
        try:
            exec(compile(rendered, "<fixed-transport>", "exec"), module.__dict__)
            return module.__dict__
        finally:
            sys.modules.pop(module.__name__, None)

    def test_exact_raw_bundle_executes_and_all_literal_interpreter_prefixes_fit_linux(self):
        result = self.execute(self.rendered)
        self.assertEqual(result["_s"], self.source)
        self.assertTrue(callable(result["main"]))
        self.assertEqual(result["NR_PACKAGE_SHA256"], hashlib.sha256(native_package()[0]).hexdigest())
        for prefix in ("python3 -I -c ", "python3 -I -B -S -c ", "/usr/bin/python3.11 -I -B -S -c "):
            command = prefix+shlex.quote(self.rendered)
            self.assertLess(len(command.encode()), 96*1024)
            self.assertLess(len(command.encode()), 131072)
            self.assertEqual(shlex.split(command)[-1], self.rendered)

    def test_foreign_source_and_decompressed_or_compressed_oversize_are_rejected(self):
        for source in (b"print('foreign source')", b"x"*(SOURCE_LIMIT+1)):
            with self.assertRaises(ValueError):
                worker_command_source(source)
        for payload in (zlib.compress(b"x"*(SOURCE_LIMIT+1)), b"x"*(64*1024+1)):
            rendered = self.rendered.replace(repr(self.encoded), repr(base64.b64encode(payload).decode()), 1)
            with self.assertRaises((ValueError, zlib.error)):
                self.execute(rendered)

    def test_truncated_trailing_hash_mismatch_and_invalid_base64_fail_before_execution(self):
        compressed = base64.b64decode(self.encoded)
        for body in (compressed[:-1], compressed+b"trailing", zlib.compress(self.source[:-1])):
            changed = self.rendered.replace(repr(self.encoded), repr(base64.b64encode(body).decode()), 1)
            with self.assertRaises(ValueError):
                self.execute(changed)
        changed = self.rendered.replace(hashlib.sha256(self.source).hexdigest(), "f"*64)
        with self.assertRaises(ValueError):
            self.execute(changed)
        changed = self.rendered.replace(repr(self.encoded), repr("invalid!"), 1)
        with self.assertRaises(ValueError):
            self.execute(changed)
