"""Smoke checks for the installed public command."""

import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "cineforge", "--version"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "cineforge 0.1.0")

    def test_unknown_argument_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "cineforge", "--render"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
