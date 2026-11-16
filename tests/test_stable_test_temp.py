"""Exercise stable fixture selection in fresh interpreters with real permissions."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StableTestTempTests(unittest.TestCase):
    def setUp(self):
        self.sim = Path(tempfile.mkdtemp(prefix="cf-stable-sim-")).resolve()
        self.addCleanup(self.cleanup_sim)
        self.blocked = self.sim / "blocked"
        self.blocked.mkdir()
        self.inner = self.blocked / "inner"
        self.inner.mkdir()
        self.home = self.sim / "home"
        self.home.mkdir()
        self.blocked.chmod(0o555)
        self.env = dict(os.environ, TMPDIR=str(self.inner), HOME=str(self.home),
                        PYTHONPATH="src:tests", PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("XDG_CACHE_HOME", None)

    def cleanup_sim(self):
        for path in (self.blocked, self.home):
            path.chmod(0o755)
        self.sim.chmod(0o755)
        shutil.rmtree(self.sim)

    def run_python(self, code):
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT,
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_sibling_unavailable_uses_cache_and_native_read_succeeds(self):
        self.run_python('''
            import tempfile
            import unittest
            from pathlib import Path
            from stable_test_temp import stable_temporary_directory
            with stable_temporary_directory(prefix="cf-cache-") as name:
                root = Path(name).resolve()
                assert not root.is_relative_to(Path(tempfile.gettempdir()).resolve())
                assert root.is_relative_to(Path.home() / ".cache")
                assert root.parent.stat().st_mode & 0o777 == 0o700
            suite = unittest.defaultTestLoader.loadTestsFromName(
                "test_native_capabilities.FixedFileMeasurementTests."
                "test_shared_tmpdir_churn_during_read_succeeds_from_stable_fixture")
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            assert result.testsRun == 1 and result.wasSuccessful()
        ''')

    def test_unavailable_home_or_xdg_cache_fails_loudly_and_can_retry(self):
        self.home.chmod(0o555)
        for configured in (None, str(self.home)):
            with self.subTest(xdg_cache_home=configured):
                if configured is not None:
                    self.env["XDG_CACHE_HOME"] = configured
                self.run_python('''
                    import os
                    import tempfile
                    from pathlib import Path
                    import stable_test_temp as stable
                    for attempt in range(2):
                        try:
                            stable.stable_temporary_directory(prefix="cf-must-fail-")
                        except RuntimeError as exc:
                            assert "attempted:" in str(exc)
                            assert str(Path(tempfile.gettempdir()).resolve().parent) in str(exc)
                            assert "cache" in str(exc)
                        else:
                            raise AssertionError("unavailable stable locations must fail loudly")
                        assert stable._parent is None
                    Path.home().chmod(0o755)
                    try:
                        with stable.stable_temporary_directory(prefix="cf-retry-") as name:
                            assert not Path(name).resolve().is_relative_to(
                                Path(tempfile.gettempdir()).resolve())
                    finally:
                        Path.home().chmod(0o555)
                ''')

    def test_cache_inside_shared_tmpdir_is_rejected(self):
        self.env["XDG_CACHE_HOME"] = str(self.inner / "cache")
        self.run_python('''
            import stable_test_temp as stable
            try:
                stable.stable_temporary_directory(prefix="cf-must-fail-")
            except RuntimeError as exc:
                assert "inside shared TMPDIR" in str(exc)
                assert stable._parent is None
            else:
                raise AssertionError("cache inside shared TMPDIR must be refused")
        ''')

    def test_absolute_xdg_cache_preserves_existing_mode(self):
        self.env["XDG_CACHE_HOME"] = str(self.home)
        self.home.chmod(0o750)
        self.run_python('''
            import os
            from pathlib import Path
            from stable_test_temp import stable_temporary_directory
            cache = Path(os.environ["XDG_CACHE_HOME"])
            with stable_temporary_directory(prefix="cf-xdg-") as name:
                assert Path(name).parent.parent == cache
                assert cache.stat().st_mode & 0o777 == 0o750
        ''')
