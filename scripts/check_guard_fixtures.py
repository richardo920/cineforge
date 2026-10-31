"""Exercise installed CI tools against synthetic bad inputs without network access."""

import subprocess
import sys
import tempfile
from pathlib import Path


def rejected(command: list[str], cwd: Path, expected: str) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode == 0 or expected not in result.stdout + result.stderr:
        raise AssertionError(f"guard did not reject fixture as expected: {command[:4]}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory)
        bad = fixture / "invalid.py"
        bad.write_text("import os\nanswer: int = 'wrong'\n")
        rejected([sys.executable, "-m", "ruff", "check", "--isolated", str(bad)], fixture, "F401")
        rejected([sys.executable, "-m", "mypy", "--strict", str(bad)], fixture, "[assignment]")
        print("lint/type rejection fixtures: PASS")

        # Deliberately invalid build configuration exercises the actual pinned builder.
        (fixture / "pyproject.toml").write_text("[build-system]\nrequires = 42\n")
        rejected([sys.executable, "-m", "build", "--no-isolation", "--wheel"], fixture, "requires")
        print("package rejection fixture: PASS")

        subprocess.run(["git", "init", "-q"], cwd=fixture, check=True)
        # An invalid fake PEM marker is recognizable but cannot be used as a private key.
        marker = "-----BEGIN " + "RSA PRIVATE KEY-----"
        (fixture / "synthetic.txt").write_text(marker + "\nnot-a-real-key\n")
        subprocess.run(["git", "add", "synthetic.txt"], cwd=fixture, check=True)
        (fixture / "synthetic.txt").write_text("clean working copy cannot hide staged marker\n")
        rejected([sys.executable, str(root / "scripts/ci_guard.py"), "secrets",
                  "--root", str(fixture)], fixture, "potential secret detected")
        print("secret rejection fixture: PASS (invalid synthetic key only)")

        (fixture / "oversized.txt").write_bytes(b"x" * (1024 * 1024 + 1))
        subprocess.run(["git", "add", "oversized.txt"], cwd=fixture, check=True)
        rejected([sys.executable, str(root / "scripts/ci_guard.py"), "artifacts",
                  "--root", str(fixture)], fixture, "exceeds")
        print("artifact-size rejection fixture: PASS")

        # An unreviewed pin fails even when syntactically valid and never gets installed.
        (fixture / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools==1.0"]\n[project]\n'
            'dependencies = ["unreviewed-package==1.0"]\n')
        (fixture / "requirements-ci.txt").write_text("")
        (fixture / "config").mkdir()
        (fixture / "config/dependency-policy.json").write_text(
            '{"runtime": {}, "build": {"setuptools": "1.0"}, "ci": {}}')
        rejected([sys.executable, str(root / "scripts/ci_guard.py"), "dependencies",
                  "--root", str(fixture)], fixture, "differ from reviewed exact pins")
        print("dependency rejection fixture: PASS")


if __name__ == "__main__":
    main()
