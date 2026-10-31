"""Policy failures use generated metadata, never real secrets or production assets."""

import copy
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "ci_guard", Path(__file__).resolve().parents[1] / "scripts" / "ci_guard.py",
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ArtifactTests(unittest.TestCase):
    def test_regular_source_and_exact_size_boundary(self):
        self.assertEqual(guard.check_blob("src/code.py", guard.MAX_BLOB_BYTES, "100644"), [])
        self.assertTrue(guard.check_blob("data.bin", guard.MAX_BLOB_BYTES + 1, "100644"))

    def test_private_media_credentials_and_external_entries_rejected(self):
        for path, mode in [("image.PNG", "100644"), ("nested/private/a.json", "100644"),
                           (".env.production", "100644"), ("link", "120000"),
                           ("vendor", "160000")]:
            with self.subTest(path=path):
                self.assertTrue(guard.check_blob(path, 1, mode))
        self.assertEqual(guard.check_blob(".env.example", 1, "100644"), [])

    def test_staged_large_blob_cannot_hide_behind_small_working_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            target = root / "oversized.txt"
            target.write_bytes(b"x" * (guard.MAX_BLOB_BYTES + 1))
            subprocess.run(["git", "add", "oversized.txt"], cwd=root, check=True)
            target.write_text("small working copy")
            self.assertIn("exceeds", " ".join(guard.tracked_artifacts(root)))


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.project = {"project": {"dependencies": []},
                        "build-system": {"requires": ["builder==1.0"]}}
        self.policy = {"runtime": {}, "build": {"builder": "1.0"}, "ci": {"tool": "2.0"}}

    def test_approved_pins_pass(self):
        self.assertEqual(guard.dependencies(self.project, ["tool==2.0"], self.policy), [])

    def test_unpinned_changed_url_or_unreviewed_dependencies_fail(self):
        for dependency in ["new-package==1.0", "new-package>=1", "pkg @ https://example.invalid/x"]:
            project = copy.deepcopy(self.project)
            project["project"]["dependencies"] = [dependency]
            with self.subTest(dependency=dependency):
                self.assertTrue(guard.dependencies(project, ["tool==2.0"], self.policy))
        self.assertTrue(guard.dependencies(self.project, ["tool==3.0"], self.policy))
        self.assertTrue(guard.dependencies(self.project, ["tool==2.0", "Tool==2.0"], self.policy))

    def test_dynamic_and_optional_dependencies_require_review(self):
        for field, value in [("dynamic", ["dependencies"]),
                             ("optional-dependencies", {"extra": ["new==1.0"]})]:
            project = copy.deepcopy(self.project)
            project["project"][field] = value
            self.assertTrue(guard.dependencies(project, ["tool==2.0"], self.policy))


if __name__ == "__main__":
    unittest.main()
