"""Local credential configuration stays tied to the actual controller reader."""

import ast
import inspect
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cineforge.cloud_smoke import CloudSmokeError, external_api_key


class EnvironmentConfigTests(unittest.TestCase):
    def test_example_assignments_have_a_supported_controller_reader(self):
        example = Path(__file__).resolve().parents[1] / ".env.example"
        assignments = [line.split("=", 1)[0] for line in example.read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
        reader = ast.parse(inspect.getsource(external_api_key))
        read_keys = {node.args[0].value for node in ast.walk(reader)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and ast.unparse(node.func) == "os.environ.get"
                     and isinstance(node.args[0], ast.Constant)}
        self.assertEqual(set(assignments), read_keys)
        self.assertEqual(len(assignments), len(set(assignments)))
        self.assertNotIn("RUNPOD_POD_ID", assignments)

    def test_environment_fallback_validation_and_private_file_precedence(self):
        variable = "RUNPOD_API_KEY"
        with patch.dict(os.environ, {variable: "fixture-environment-key"}, clear=True):
            self.assertEqual(external_api_key(), "fixture-environment-key")
            with TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                root.chmod(0o700)
                key = root / "key"
                key.write_text("fixture-file-key\n")
                key.chmod(0o600)
                self.assertEqual(external_api_key(str(key)), "fixture-file-key")
                key.write_text("invalid key")
                with self.assertRaises(CloudSmokeError):
                    external_api_key(str(key))  # Never fall back after a bad explicit file.
        for value in (None, "", "short", "invalid key"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["RUNPOD_API_KEY"] = value
                with self.assertRaises(CloudSmokeError):
                    external_api_key()
