"""Static M5 lease/service scope, independent of not-yet-generated speech or graphs.

Scope records bind separately operator-verified approval; they never authenticate an
owner, load code from JSON, prove runtime fit, or approve a materialized S2V request.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineforge import s2v_profile as profile
from cineforge.cloud_budget import BudgetApproval, CloudBudgetError
from cineforge.devices import ModelSpec
from cineforge.rights import EvidenceRef
from cineforge.workflows import canonical


def _scope(approval: S2VBudgetApproval) -> dict[str, Any]:
    return _scope_fields(approval.journal_directory, approval.region, approval.operator_receipt,
                         approval.aggregate_usd, approval.lifetime_seconds,
                         approval.hourly_ceiling_usd)


def _scope_fields(directory: str, region: str, receipt: EvidenceRef, aggregate: float,
                  lifetime: float, hourly: float) -> dict[str, Any]:
    return {"task_id": "M5", "recipe_sha256": profile.recipe_sha256(),
        "operator_receipt": receipt.to_dict(),
        "image": profile.IMAGE, "gpu_id": profile.GPU_ID, "region": region,
        "journal_directory": directory, "min_ram_per_gpu_gb": 64,
        "aggregate_usd": aggregate, "max_attempts": 1, "max_concurrent": 1,
        "lifetime_seconds": lifetime, "hourly_ceiling_usd": hourly, "disk_gb": 80,
        "min_cuda": "12.8", "ports": ["22/tcp"], "volumes": [], "region_fallback": False,
        "native_speech": "separately reviewed installed-source/weight/receipt gate under same guard"}


@dataclass(frozen=True)
class S2VBudgetApproval(BudgetApproval):
    """Closed M5 scope; static recipe/resources only, not an unknown approved graph."""
    journal_directory: str
    operator_receipt: EvidenceRef

    def __post_init__(self) -> None:
        if (type(self.operator_receipt) is not EvidenceRef
                or not isinstance(self.journal_directory, str)
                or not Path(self.journal_directory).is_absolute()
                or str(Path(self.journal_directory)) != self.journal_directory
                or ".." in Path(self.journal_directory).parts
                or not self.journal_directory.isprintable()
                or any(char in self.journal_directory for char in "\\:%")
                or type(self.region) is not str or re.fullmatch(r"[A-Z0-9-]{3,32}", self.region) is None
                or type(self.aggregate_usd) not in (int, float) or not 0 < self.aggregate_usd <= 5
                or type(self.lifetime_seconds) not in (int, float)
                or not 120 < self.lifetime_seconds <= 1800
                or type(self.hourly_ceiling_usd) not in (int, float)
                or not 0 < self.hourly_ceiling_usd <= 1.25
                or self.image != profile.IMAGE or self.gpu_id != profile.GPU_ID
                or self.min_cuda != "12.8" or type(self.disk_gb) is not int or self.disk_gb != 80
                or type(self.max_attempts) is not int or self.max_attempts != 1
                or type(self.max_concurrent) is not int or self.max_concurrent != 1
                or self.model != profile.model()
                or self.model_workflow_sha256 != profile.recipe_sha256()
                or self.subject_sha256 != hashlib.sha256(canonical(_scope(self))).hexdigest()):
            raise CloudBudgetError("invalid closed M5 recipe budget scope")

    @classmethod
    def for_scope(cls, journal_directory: str, region: str, operator_receipt: EvidenceRef,
                  *, aggregate_usd: float = 5, lifetime_seconds: float = 1800,
                  hourly_ceiling_usd: float = 1.25) -> S2VBudgetApproval:
        # Build once from independently selected operator arguments, never creative JSON.
        if type(operator_receipt) is not EvidenceRef:
            raise CloudBudgetError("M5 budget requires separately verified operator evidence")
        digest = None
        try:
            digest = hashlib.sha256(canonical(_scope_fields(journal_directory, region,
                operator_receipt, aggregate_usd, lifetime_seconds, hourly_ceiling_usd))).hexdigest()
        except (ValueError, TypeError, OverflowError):
            pass
        if digest is None:
            raise CloudBudgetError("invalid M5 scope declarations")
        return cls(digest, profile.recipe_sha256(), profile.model(), profile.IMAGE, profile.GPU_ID,
            region, "12.8", aggregate_usd, 1, 1, lifetime_seconds, hourly_ceiling_usd, 80,
            journal_directory, operator_receipt)

    @property
    def min_ram_per_gpu_gb(self) -> int:
        return 64


@dataclass(frozen=True)
class S2VServiceContext:
    """Exact nonserialized service context used before any native speech bytes exist."""
    approval: S2VBudgetApproval

    def preflight(self) -> None:
        if type(self.approval) is not S2VBudgetApproval:
            raise CloudBudgetError("M5 service requires its separately checked budget scope")
        self.approval.__post_init__()

    @property
    def model(self) -> ModelSpec:
        self.preflight()
        return profile.model()

    @property
    def declarations(self) -> dict[str, Any]:
        self.preflight()
        return profile.service_declarations()

    @property
    def worker_profile(self) -> dict[str, str]:
        self.preflight()
        return {"id": profile.PROFILE_ID, "scope_sha256": self.approval.subject_sha256}


def prepare_service_journal(context: S2VServiceContext) -> None:
    """Offline initialization only; a retained journal is never reset or refunded."""
    from cineforge.cloud_journal import CloudJournal
    from cineforge.workflows import strict_object
    if type(context) is not S2VServiceContext:
        raise CloudBudgetError("M5 journal requires its checked service scope")
    context.preflight()
    with CloudJournal(context.approval.journal_directory) as journal:
        if journal.initial is None:
            journal.checkpoint(canonical({"schema_version": 1,
                "approval_sha256": context.approval.subject_sha256,
                "last_wall": time.time(), "records": {}}))
        elif strict_object(journal.initial).get("approval_sha256") != context.approval.subject_sha256:
            raise CloudBudgetError("existing M5 journal belongs to a different scope")
