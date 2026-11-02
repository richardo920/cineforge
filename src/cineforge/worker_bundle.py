"""Bundle only fixed installed worker sources; JSON inputs never supply code or module paths."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from cineforge.private_store import read_private_input

_PROFILE_IMPORT = b"from cineforge.worker_profiles import I2V_WORKER, T2V_WORKER, WorkerProfile\n"
_FUTURE = b"from __future__ import annotations\n"
_MAIN = b'\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
MAX_SOURCE_BYTES = 128 * 1024


class WorkerBundleError(ValueError):
    """The fixed local worker sources are missing, incomplete or unsupported."""


def worker_source() -> bytes:
    """Deterministic single-file stdlib bundle, independent of the remote module search path."""
    failed = False
    result = b""
    try:
        root = Path(__file__).parent
        profiles = read_private_input(root / "worker_profiles.py", max_bytes=16 * 1024)
        bootstrap = read_private_input(root / "worker_bootstrap.py", max_bytes=MAX_SOURCE_BYTES)
        if (not profiles.endswith(b"\n# END_REVIEWED_WORKER_PROFILES\n")
                or bootstrap.count(_PROFILE_IMPORT) != 1 or bootstrap.count(_FUTURE) != 1
                or not bootstrap.endswith(_MAIN)):
            raise ValueError
        definitions = ast.parse(profiles).body
        classes = {node.name for node in definitions if isinstance(node, ast.ClassDef)}
        assignments = {target.id for node in definitions if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        if (not {"ReferenceRuntime", "WorkerProfile"} <= classes
                or not {"T2V_WORKER", "I2V_WORKER"} <= assignments):
            raise ValueError
        result = (_FUTURE + profiles + b"\n"
                  + bootstrap.replace(_FUTURE, b"").replace(_PROFILE_IMPORT, b""))
        if len(result) > MAX_SOURCE_BYTES:
            raise ValueError
        tree = ast.parse(result)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or node.module is None:
                    raise ValueError
                modules = [node.module]
            else:
                continue
            if any(module.split(".")[0] not in sys.stdlib_module_names for module in modules):
                raise ValueError
        compile(tree, "<reviewed-worker-bundle>", "exec")
    except (OSError, ValueError, TypeError, SyntaxError, RecursionError):
        failed = True
    if failed:
        raise WorkerBundleError("fixed reviewed worker source bundle unavailable")
    return result
