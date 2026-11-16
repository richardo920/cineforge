"""Temporary fixtures isolated from shared temporary-directory churn."""

import atexit
import os
from pathlib import Path
import secrets
import shutil
import tempfile


_parent = None
_initialized = False


def stable_temporary_directory(*, prefix):
    global _parent, _initialized
    # read_native_file snapshots every ancestor from / and refuses any mtime/ctime
    # change. Real native-read fixtures must avoid shared TMPDIR churn from peers.
    if not _initialized:
        try:
            parent = Path(tempfile.gettempdir()).resolve().parent
            private = parent / f"cineforge-test-{os.getpid()}-{secrets.token_hex(8)}"
            private.mkdir(mode=0o700)
        except OSError:
            pass  # e.g. unprivileged Linux cannot create a sibling of /tmp.
        else:
            _parent = private
            atexit.register(shutil.rmtree, private, ignore_errors=True)
        _initialized = True
    try:
        return tempfile.TemporaryDirectory(prefix=prefix, dir=_parent)
    except OSError:
        return tempfile.TemporaryDirectory(prefix=prefix)
