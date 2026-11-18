"""Explicit trusted operator composition, separate from untrusted creative JSON."""
from __future__ import annotations

import argparse
import hashlib
import runpy
import time
from pathlib import Path
from typing import Any

from cineforge.a20_driver import A20Bindings, run_a20
from cineforge.a20_inputs import A20Error, CreativeInputs, FrozenInputs
from cineforge.private_store import read_private_input
from cineforge.workflows import strict_object


def command(args: argparse.Namespace) -> dict[str, Any]:
    # Start the original cap before local input review/import; never renew it.
    deadline = time.monotonic() + 7200  # Maximum approval lifetime.
    value = strict_object(read_private_input(args.inputs))
    if any(type(v) is not str for v in value.values()):
        raise A20Error("A20_INPUT_PATHS")
    creative = CreativeInputs({name: Path(path) for name, path in value.items()})
    creative.read()  # fixed missing-input name, before trusted code or credentials
    if args.allow_live is not True:
        raise A20Error("A20_LIVE_OPT_IN_REQUIRED")
    root, controller = Path(args.controller_root).absolute(), Path(args.controller).absolute()
    if (not controller.is_relative_to(root) or controller.suffix != ".py"
            or hashlib.sha256(read_private_input(controller)).hexdigest() != args.controller_sha256):
        raise A20Error("A20_CONTROLLER_IDENTITY")
    frozen = FrozenInputs.capture(creative, (Path(__file__).parent, root))
    # This is explicitly operator-supplied Python, like calling run_a20 from Python;
    # creative JSON cannot select modules, verifiers, callbacks, or executable code.
    namespace = runpy.run_path(str(controller), run_name="cineforge_a20_operator")
    frozen.check()
    factory = namespace.get("build_bindings")
    if not callable(factory):
        raise A20Error("A20_CONTROLLER_FACTORY")
    bindings = factory(creative)
    if type(bindings) is not A20Bindings:
        raise A20Error("A20_CONTROLLER_BINDINGS")
    frozen.check()
    report = run_a20(bindings, frozen, Path(args.output_dir),
                     absolute_deadline=deadline, allow_live=args.allow_live)
    result = {key: report[key] for key in
              ("outcome", "release_verified", "human_review", "export_authority")}
    if "retained_node" in report:
        result["retained_node"] = report["retained_node"]
    return result
