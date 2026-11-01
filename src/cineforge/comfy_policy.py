"""Reviewed configuration and operator-attested worker admission; no control-plane I/O."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cineforge.comfy import SUPPORTED_SERVER_REVISION
from cineforge.contracts import Shot
from cineforge.devices import Backend, GPUDevice, ModelSpec, Precision, select_device
from cineforge.private_store import read_private_input
from cineforge.providers import ExecutionRequest, ProviderError
from cineforge.selection import require_selection_approval
from cineforge.workflows import (
    InputBinding,
    WorkflowTemplate,
    canonical,
    strict_object,
    workflow_links,
)

CORE_CLASSES = frozenset({
    "CLIPLoader", "CLIPTextEncode", "EmptyHunyuanLatentVideo", "KSampler",
    "ModelSamplingSD3", "SaveWEBM", "UNETLoader", "VAEDecode", "VAELoader",
})


class ComfyPolicyError(ValueError):
    """Invalid reviewed configuration or missing operator verification."""


def safe_relative_path(value: object) -> str:
    if (not isinstance(value, str) or not value or not value.isprintable()
            or any(c in value for c in "\\:%")
            or any(not part or part in (".", "..") for part in value.split("/"))):
        raise ComfyPolicyError("unsafe configuration path")
    return value


def _digest(value: object, length: int = 64) -> None:
    if (not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value)
            or value == "0" * length):
        raise ComfyPolicyError("missing immutable digest")


@dataclass(frozen=True)
class ApprovedComfySelection:
    """A separately trusted operator-selected ledger, never supplied by a creative plan.

    Approval hashes bind declarations; they do not authenticate the owner/receipt.
    The operator must verify provenance of this configuration before using it.
    """

    selection_json: str
    graph_json: str
    license_bytes: bytes

    def __post_init__(self) -> None:
        failed = False
        try:
            if not isinstance(self.license_bytes, bytes):
                raise ComfyPolicyError("license must be immutable bytes")
            selection, graph = strict_object(self.selection_json), strict_object(self.graph_json)
            if type(selection.get("schema_version")) is not int or selection["schema_version"] != 1:
                raise ComfyPolicyError("unsupported selection schema")
            require_selection_approval(selection, graph)
            runner, workflow, model = (selection[k] for k in ("runner", "workflow", "model"))
            if (runner["revision"] != SUPPORTED_SERVER_REVISION or runner["custom_nodes"] != []
                    or workflow["output_kind"] != "images" or workflow["output_index"] != 0
                    or workflow["media_type"] != "video/webm"
                    or hashlib.sha256(self.license_bytes).hexdigest() != workflow["license_sha256"]):
                raise ComfyPolicyError("unsupported reviewed runner or workflow")
            for revision in (runner["revision"], model["revision"], workflow["source_revision"]):
                _digest(revision, 40)
            _digest(workflow["canonical_sha256"])
            classes = {node["class_type"] for node in graph.values()}
            if (not classes <= CORE_CLASSES or set(workflow["allowed_classes"]) != classes
                    or len(model["files"]) != 3):
                raise ComfyPolicyError("unapproved node classes or model files")
            assets = {}
            for asset in model["files"]:
                path = safe_relative_path(asset["path"])
                _digest(asset["sha256"])
                if (asset["revision"] != model["revision"] or path in assets
                        or type(asset["size_bytes"]) is not int or asset["size_bytes"] <= 0
                        or asset["license"] not in ("Apache-2.0", "Apache-2.0 (repository declaration)")):
                    raise ComfyPolicyError("invalid approved model asset")
                url = urlsplit(asset["url"])
                expected = f"/{model['distribution']}/resolve/{model['revision']}/{path}"
                if (url.scheme != "https" or url.netloc != "huggingface.co" or url.path != expected
                        or url.query or url.fragment):
                    raise ComfyPolicyError("model asset URL differs from pinned source")
                assets[path] = asset
            loaders = {"UNETLoader": ("unet_name", "diffusion_models"),
                       "CLIPLoader": ("clip_name", "text_encoders"),
                       "VAELoader": ("vae_name", "vae")}
            used = []
            for node in graph.values():
                if node["class_type"] in loaders:
                    field, folder = loaders[node["class_type"]]
                    name = safe_relative_path(node["inputs"][field])
                    if "/" in name:
                        raise ComfyPolicyError("model loader must name one approved file")
                    used.append(f"split_files/{folder}/{name}")
            if sorted(used) != sorted(assets):
                raise ComfyPolicyError("workflow loaders differ from approved model files")
            output = graph[workflow["output_node"]]
            prefix = safe_relative_path(output["inputs"]["filename_prefix"])
            if output["class_type"] != "SaveWEBM" or "/" in prefix:
                raise ComfyPolicyError("workflow output must use a private relative prefix")
            bindings = workflow["bindings"]
            for key, kind, field in (("prompt", "CLIPTextEncode", "text"),
                                     ("seed", "KSampler", "seed")):
                binding = bindings[key]
                if (binding["input_name"] != field
                        or graph[binding["node_id"]]["class_type"] != kind):
                    raise ComfyPolicyError("unsafe workflow request binding")
            self._links(graph)
            object.__setattr__(self, "selection_json", canonical(selection).decode())
            object.__setattr__(self, "graph_json", canonical(graph).decode())
            self.workflow  # Validate the complete execution template before any network I/O.
            self.model
        except (KeyError, TypeError, ValueError, AttributeError, RecursionError):
            failed = True
        if failed:
            raise ComfyPolicyError("invalid or unapproved ComfyUI selection")

    @staticmethod
    def _links(graph: dict[str, Any]) -> None:
        try:
            workflow_links(graph)
        except ProviderError:
            raise ComfyPolicyError("invalid workflow links") from None

    @property
    def declarations(self) -> dict[str, Any]:
        return strict_object(self.selection_json)

    @property
    def model(self) -> ModelSpec:
        data = self.declarations["model"]
        return ModelSpec(data["model_id"], data["revision"], data["task"],
                         frozenset(Backend(b) for b in data["allowed_backends"]),
                         data["minimum_memory_mib"], data["license"],
                         Precision(data["required_precision"]), data["commercial_use"])

    @property
    def workflow(self) -> WorkflowTemplate:
        data = self.declarations["workflow"]
        return WorkflowTemplate(
            self.graph_json, self.declarations["model"]["model_id"],
            self.declarations["model"]["revision"], frozenset(data["allowed_classes"]),
            InputBinding(**data["bindings"]["prompt"]), InputBinding(**data["bindings"]["seed"]),
            data["output_node"], data["output_kind"], data["artifact_license"],
            data["output_index"], data["duration_seconds"],
        )

    def request(self, plan_shot: Shot) -> ExecutionRequest:
        if plan_shot.character_revision is not None:
            raise ComfyPolicyError("T2V selection cannot condition a character reference")
        request = ExecutionRequest(plan_shot, self.model)
        if (len(plan_shot.prompt) > 8192
                or any(ord(c) < 32 and c not in "\n\r\t" for c in plan_shot.prompt)
                or re.search(r"\bembedding\s*:", plan_shot.prompt, re.IGNORECASE)):
            raise ComfyPolicyError("unsupported prompt controls or external embedding reference")
        self.workflow.translate(request)
        return request


def load_comfy_selection(path: str | Path) -> ApprovedComfySelection:
    """Choose reviewed configuration separately from creative input; do not download anything."""
    failed = False
    result = None
    try:
        path = Path(path)
        selection = strict_object(read_private_input(path))
        graph_name = safe_relative_path(selection["workflow"]["path"])
        license_name = safe_relative_path(selection["workflow"]["license_path"])
        if "/" in graph_name or "/" in license_name:
            raise ComfyPolicyError("selection references must be sibling files")
        result = ApprovedComfySelection(canonical(selection).decode(),
                                       read_private_input(path.parent / graph_name).decode("utf-8"),
                                       read_private_input(path.parent / license_name))
    except (OSError, ValueError, KeyError, TypeError):
        failed = True
    if failed or result is None:
        raise ComfyPolicyError("could not load trusted reviewed selection")
    return result


@dataclass(frozen=True)
class WorkerVerification:
    """Operator attestations, not remote inspection or cryptographic identity proof."""

    record_json: str

    def __post_init__(self) -> None:
        failed = False
        try:
            data = strict_object(self.record_json)
            names = {"schema_version", "operator_verified", "worker_id", "server_revision",
                     "image_digest", "python_version", "torch_version", "cuda_version", "gpu",
                     "model_files", "custom_nodes_disabled"}
            if (set(data) != names or type(data["schema_version"]) is not int
                    or data["schema_version"] != 1 or data["operator_verified"] is not True
                    or data["custom_nodes_disabled"] is not True
                    or data["server_revision"] != SUPPORTED_SERVER_REVISION
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", data["worker_id"])):
                raise ComfyPolicyError("incomplete worker verification")
            if not data["image_digest"].startswith("sha256:"):
                raise ComfyPolicyError("worker image must be pinned by digest")
            _digest(data["image_digest"][7:])
            for name in ("python_version", "torch_version", "cuda_version"):
                if not re.fullmatch(r"[A-Za-z0-9.+_-]{1,80}", data[name]):
                    raise ComfyPolicyError("invalid worker runtime version")
            if set(data["gpu"]) != {"name", "backend", "memory_mib"}:
                raise ComfyPolicyError("malformed worker GPU verification")
            object.__setattr__(self, "record_json", canonical(data).decode())
        except (ValueError, TypeError, KeyError, AttributeError):
            failed = True
        if failed:
            raise ComfyPolicyError("invalid operator worker verification")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerVerification:
        return cls(canonical(value).decode())

    @property
    def record(self) -> dict[str, Any]:
        return strict_object(self.record_json)

    def require_matches(self, selection: ApprovedComfySelection) -> None:
        failed = False
        try:
            data, declarations = self.record, selection.declarations
            gpu = data["gpu"]
            if gpu["name"] != declarations["hardware"]["gpu"]:
                raise ComfyPolicyError("worker differs from approved hardware")
            select_device(selection.model, [GPUDevice(
                data["worker_id"], Backend(gpu["backend"]), gpu["memory_mib"],
                frozenset({selection.model.required_precision}),
            )])
            expected = [{k: asset[k] for k in ("path", "sha256", "size_bytes")}
                        for asset in declarations["model"]["files"]]
            if canonical(data["model_files"]) != canonical(expected):
                raise ComfyPolicyError("worker model bytes differ from approved files")
        except (ValueError, KeyError, TypeError):
            failed = True
        if failed:
            raise ComfyPolicyError("worker does not match approved selection")
