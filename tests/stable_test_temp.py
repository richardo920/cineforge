"""Temporary fixtures isolated from shared temporary-directory churn."""

import atexit
import os
import secrets
import shutil
import tempfile
from pathlib import Path

_parent = None


def _cache_directory():
    configured = os.environ.get("XDG_CACHE_HOME", "")
    return Path(configured) if Path(configured).is_absolute() else Path.home() / ".cache"


def stable_temporary_directory(*, prefix):
    global _parent
    shared = Path(tempfile.gettempdir()).resolve()
    # Native reads snapshot every ancestor; shared TMPDIR churn must stay outside
    # the fixture's ancestry, including when /tmp's parent is not writable.
    if _parent is None:
        attempted = []
        for label, location in (
            (str(shared.parent), lambda: shared.parent),
            ("user cache ($XDG_CACHE_HOME or ~/.cache)", _cache_directory),
        ):
            private = None
            created = False
            try:
                directory = location()
                private = directory / f"cineforge-test-{os.getpid()}-{secrets.token_hex(8)}"
                attempted.append(str(private))
                directory.mkdir(parents=True, exist_ok=True)
                private.mkdir(mode=0o700)
                created = True
                resolved = private.resolve()
                if resolved.is_relative_to(shared):
                    raise RuntimeError(f"stable fixture parent is inside shared TMPDIR: {resolved}")
            except (OSError, RuntimeError) as exc:
                if created:
                    shutil.rmtree(private, ignore_errors=True)
                attempted.append(f"{label}: {exc}")
                continue
            _parent = resolved
            atexit.register(shutil.rmtree, resolved, ignore_errors=True)
            break
        else:
            raise RuntimeError("cannot create stable test directory outside shared TMPDIR; attempted: "
                               + "; ".join(attempted))
    if _parent.is_relative_to(shared):
        raise RuntimeError(f"stable fixture parent is inside shared TMPDIR: {_parent}")
    temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=_parent)
    if Path(temporary.name).resolve().is_relative_to(shared):
        temporary.cleanup()
        raise RuntimeError("stable fixture child is inside shared TMPDIR")
    return temporary
