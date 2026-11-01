"""Pure approval-scope checks; receipt authentication and live admission stay external."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


class SelectionError(ValueError):
    """A selection is missing approval or differs from the approved subject."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def selection_subject(selection: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable subject an owner approval must bind; perform no I/O."""
    try:
        model, workflow, runner = (selection[k] for k in ("model", "workflow", "runner"))
        hardware, approval = selection["hardware"], selection["approval"]
        return {
            "model": {k: model[k] for k in ("model_id", "revision", "distribution", "task",
                                          "allowed_backends", "minimum_memory_mib",
                                          "required_precision", "license", "commercial_use")},
            "weights": [{k: asset[k] for k in ("path", "revision", "sha256", "size_bytes",
                                                    "url", "license")}
                        for asset in model["files"]],
            "runner": {k: runner[k] for k in ("repository", "revision", "custom_nodes")},
            "workflow": {k: workflow[k] for k in (
                "canonical_sha256", "allowed_classes", "bindings", "output_node", "output_kind",
                "output_index", "artifact_license", "media_type", "duration_seconds",
                "source_repository", "source_revision", "source_path", "source_raw_sha256",
                "license", "license_sha256",
            )},
            "hardware": {k: hardware[k] for k in ("gpu", "vram_gib")},
            "weight_downloads": approval["weight_downloads"],
            "spend_limit_usd": approval["spend_limit_usd"],
        }
    except (KeyError, TypeError) as exc:
        raise SelectionError("incomplete selection approval subject") from exc


def require_selection_approval(selection: dict[str, Any], graph: dict[str, Any]) -> None:
    """Check declarations match a recorded approval, not receipt authenticity or live readiness.

    The coordinator must separately verify the receipt. This function does not admit workers,
    download weights, spend budget, or establish model-byte integrity or measured compatibility.
    """
    try:
        approval = selection["approval"]
        record = approval["model_workflow"]
        if selection["state"] != "selection_approved" or record["status"] != "approved":
            raise SelectionError("model/workflow selection is not approved")
        for flag in ("model_approved", "workflow_approved", "hardware_approved"):
            if record[flag] is not True:
                raise SelectionError("incomplete selection approval scope")
        if approval["weight_downloads"] is not True:
            raise SelectionError("worker weight downloads are not approved")
        if (not isinstance(record["approved_by"], str) or not record["approved_by"].strip()
                or not isinstance(record["evidence"], str)
                or not re.fullmatch(
                    r"https://github\.com/richardo920/cineforge/issues/40#issuecomment-[0-9]+",
                    record["evidence"],
                )):
            raise SelectionError("approval receipt and approver are required")
        budget = approval["spend_limit_usd"]
        if (type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0):
            raise SelectionError("approval requires a positive finite spend limit")
        if hashlib.sha256(_canonical(graph).encode("utf-8")).hexdigest() != (
            selection["workflow"]["canonical_sha256"]
        ):
            raise SelectionError("workflow bytes differ from approved digest")
        subject_digest = hashlib.sha256(_canonical(selection_subject(selection)).encode("utf-8"))
        if subject_digest.hexdigest() != record["approved_subject_sha256"]:
            raise SelectionError("selection differs from recorded approved subject")
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, SelectionError):
            raise
        raise SelectionError("missing or malformed selection approval") from exc
