"""Explicit input bindings into a caller-reviewed, immutable ComfyUI API graph."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from cineforge.comfy_transport import positive_seconds
from cineforge.providers import ExecutionRequest, ProviderError, _text


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def strict_object(raw: bytes | str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError("nonfinite JSON number")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(value, dict):
            raise ValueError
        canonical(value)
        return value
    except (ValueError, TypeError, RecursionError):
        raise ProviderError("malformed ComfyUI JSON object") from None


@dataclass(frozen=True)
class InputBinding:
    node_id: str
    input_name: str

    def __post_init__(self) -> None:
        _text(self.node_id, "binding node")
        _text(self.input_name, "binding input")


@dataclass(frozen=True)
class WorkflowTemplate:
    """Class names are a caller-reviewed allowlist, not installation instructions.

    The digest covers the canonical API graph before prompt/seed substitution.
    Duration and other model-specific controls remain pinned in that graph.
    """

    graph_json: str
    model_id: str
    model_revision: str
    allowed_classes: frozenset[str]
    prompt: InputBinding
    seed: InputBinding
    output_node: str
    output_kind: str
    artifact_license: str
    output_index: int = 0
    duration_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "output_node", "artifact_license"):
            _text(getattr(self, name), name)
        if (not isinstance(self.graph_json, str) or len(self.graph_json) > 4 * 1024 * 1024
                or not isinstance(self.allowed_classes, frozenset) or not self.allowed_classes
                or any(not isinstance(c, str) or not c for c in self.allowed_classes)
                or not isinstance(self.prompt, InputBinding)
                or not isinstance(self.seed, InputBinding) or self.prompt == self.seed
                or self.output_kind not in ("images", "videos", "gifs")
                or type(self.output_index) is not int or not 0 <= self.output_index <= 1000):
            raise ProviderError("invalid workflow declaration")
        graph = strict_object(self.graph_json)
        positive_seconds(self.duration_seconds)
        if not graph or self.output_node not in graph:
            raise ProviderError("workflow output node is missing")
        for node in graph.values():
            if (not isinstance(node, dict) or set(node) != {"class_type", "inputs"}
                    or not isinstance(node["class_type"], str)
                    or node["class_type"] not in self.allowed_classes
                    or not isinstance(node["inputs"], dict)):
                raise ProviderError("workflow contains an undeclared node class or malformed node")
        for binding in (self.prompt, self.seed):
            if (binding.node_id not in graph
                    or binding.input_name not in graph[binding.node_id]["inputs"]):
                raise ProviderError("workflow input binding is missing")
        object.__setattr__(self, "graph_json", canonical(graph).decode("utf-8"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.graph_json.encode("utf-8")).hexdigest()

    def translate(self, request: ExecutionRequest) -> dict[str, Any]:
        if (not isinstance(request, ExecutionRequest)
                or (request.model.model_id, request.model.revision) !=
                (self.model_id, self.model_revision)
                or request.shot.workflow_digest != self.digest or request.shot.seed is None
                or request.shot.duration_seconds != self.duration_seconds):
            raise ProviderError("request must pin this workflow/model and an explicit seed")
        graph = strict_object(self.graph_json)
        graph[self.prompt.node_id]["inputs"][self.prompt.input_name] = request.shot.prompt
        graph[self.seed.node_id]["inputs"][self.seed.input_name] = request.shot.seed
        return graph
