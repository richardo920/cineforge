"""Render only the fixed first-party worker bundle within Linux's single-argument bound."""
from __future__ import annotations

import base64
import hashlib
import zlib

SOURCE_LIMIT = 160 * 1024
COMPRESSED_LIMIT = 64 * 1024
COMMAND_LIMIT = 96 * 1024


def worker_command_source(source: bytes) -> str:
    from cineforge.worker_bundle import worker_source
    if type(source) is not bytes or not 0 < len(source) <= SOURCE_LIMIT or source != worker_source():
        raise ValueError("fixed current first-party bundle required")
    compressed = zlib.compress(source, 9)
    if len(compressed) > COMPRESSED_LIMIT:
        raise ValueError("compressed worker exceeds bound")
    encoded = base64.b64encode(compressed).decode("ascii")
    # The decoder runs under the existing literal profile-specific interpreter/flags.
    # No stdin/request value supplies executable source or changes these caps.
    result = (
        "import base64,hashlib,zlib\n"
        f"_c=base64.b64decode({encoded!r},validate=True)\n"
        f"if len(_c)>{COMPRESSED_LIMIT}:raise ValueError('compressed worker bound')\n"
        "_d=zlib.decompressobj()\n"
        f"_s=_d.decompress(_c,{SOURCE_LIMIT+1})\n"
        f"if len(_s)!={len(source)} or len(_s)>{SOURCE_LIMIT} or not _d.eof or _d.unused_data or _d.unconsumed_tail:raise ValueError('worker framing')\n"
        f"if hashlib.sha256(_s).hexdigest()!={hashlib.sha256(source).hexdigest()!r}:raise ValueError('worker identity')\n"
        "exec(compile(_s,'<reviewed-worker-bundle>','exec'),globals())\n"
    )
    if len(result.encode()) > COMMAND_LIMIT:
        raise ValueError("encoded worker command exceeds bound")
    return result
