# GPU and model proposal

| Boundary | Contract |
| --- | --- |
| Provisioner | acquire(requirements, budget), inspect(), release() |
| Execution provider | submit(request, idempotency_key), status(), cancel(), result() |
| Model adapter | validate input, translate request, normalize output |
| Device | backend, VRAM, identity, supported precision |
| Model | task, pinned revision, allowed backends, memory, license |

Support CUDA, ROCm, Metal, and CPU only where the selected model/workflow declares compatibility. Start with a fake provider, then a local ComfyUI runner; cloud lifecycle adapters require bounded spend, timeouts, redacted logs, and cleanup tests.

`cineforge.devices` now provides declared `GPUDevice`/`ModelSpec` contracts and pure offline `select_device(model, devices)`. Selection requires backend, integer MiB capacity and precision compatibility; ties prefer least sufficient memory, then backend value, then device ID.

All fixtures are synthetic. Revision/license/commercial-use fields are declarations, not verified provenance or permission; this module performs no hardware detection, provisioning or execution.

`providers.py` exposes `ExecutionRequest(shot, model, attempt=1)` and `ExecutionProvider`: `submit(request, idempotency_key) -> ShotRun`, `status(job_id) -> ShotRun`, `cancel(job_id) -> ShotRun`, `result(job_id) -> ExecutionResult(run, artifact, payload)`.

`FakeExecutionProvider` submits queued jobs; each status poll advances queued → running → succeeded (`fail=True`: failed). Terminal jobs stay terminal, duplicate keys return the existing snapshot, conflicting requests fail, and only succeeded jobs return results.

Results contain synthetic JSON bytes with full shot/model provenance, actual SHA256/size, zero cost and fixed synthetic timestamps; `memory://` URIs do not write storage. IDs/bytes repeat for the same request/key across instances, but jobs are held only in memory; callers persist payloads separately. `provisioning.Provisioner` defines the provider-neutral acquire/inspect/release contract.

`ComfyExecutionProvider(transport, workflow)` implements the same protocol with bounded requests, polling, selected-output download and actual checksums; it neither provisions compute nor persists bytes. `WorkflowTemplate` pins a canonical API graph/model, reviewed node-class allowlist, prompt/seed bindings, fixed duration and one output selection; each request must match its digest/model/duration.

`ComfyHTTPTransport()` performs no network I/O until called with explicit `allow_live=True`; its default endpoint is `http://127.0.0.1:8188`, with no DNS/proxies/redirects or embedded credentials. Use an externally configured loopback tunnel for remote workers; model/hardware approval and live execution remain separate gates.

API reference: [ComfyUI server.py at 6d352802](https://github.com/comfyanonymous/ComfyUI/blob/6d352802d8f8b9133f664a1757d6c6d26390c335/server.py). Cancellation requires an explicit `verified_server_revision` equal to that full SHA and uses only `/api/jobs/{id}/cancel`, whose queue helper atomically scopes interruption; `/interrupt` and raw queue deletion are never used.

Cancellation validates the action acknowledgement and reconciles fresh history after queue disappearance; racing completion takes precedence. The API's boolean does not distinguish dequeue from interrupt, so even acknowledged queued deletion without terminal history remains uncertain; only confirmed interruption history becomes `CANCELLED`.

Idempotency lasts only for one provider instance; an uncertain submission reserves its key and cannot replay, and a job missing from both queue/history stays uncertain. Client deadline/poll exhaustion does not prove GPU work stopped: callers must explicitly cancel; completed downloads are cached, while restart recovery remains unsupported.

`load_comfy_selection` validates a separately trusted ledger/graph/license; `WorkerVerification` checks operator-attested image/runtime/GPU and exact model-file hashes. `execute_comfy` / `comfy-run` require explicit live opt-in, one matching shot and loopback transport, verify the installed core-node schemas, and persist a private atomic journal/payload outside Git; failures retain known job state and never resubmit automatically.

Selection hashes bind declarations, not owner identity; worker records are attestations, not remote inspection. Client bounds and best-effort job cancellation do not enforce cloud rental costs or guarantee worker shutdown.

Pod and Serverless will be selectable execution adapters sharing domain/model/workflow/artifact contracts. M2 implements private ComfyUI access to a manually prepared Pod; Serverless job APIs and cloud lifecycle automation remain future work.

## M3 provider review

[The owner-approved Runpod Pod scope](../../config/cloud/runpod-pod-proposal.json) compares Lambda and Modal and records a separate USD3 aggregate M3 validation cap with its [approval receipt](https://github.com/richardo920/cineforge/issues/48#issuecomment-6554788309). Live readiness remains false until M3.2/M3.3 safety gates, external credentials, explicit opt-in and fresh worker preflight pass; persistent storage, image publication and other GPU SKUs remain outside this scope.

`RunpodPodProvisioner(spec, transport, probe=..., before_create=..., event_sink=...)` reserves an `AcquisitionIntent` before POST, records each `LeaseSnapshot`, and returns a `GPUDevice` only after a separately supplied runtime probe; Runpod's `gpu.memory` is host RAM. Release verifies its own ephemeral worker's identity/configuration, deletes it, and confirms GET404; uncertain creation blocks replay, and unconfirmed cleanup retains the owned ID.

The required policy hook supplies fresh quote/approval/budget checks; the event sink persists private intent/state for M3.3 supervision. `RunpodHTTPTransport(api_key, allow_live=True)` only supplies bounded fixed-origin RESTv2 HTTPS: it does not authenticate owner approval, enforce a rental deadline, provision storage, install models, or execute workflows; a price race can still allocate before read-back rejects it. `restore_owned(intent, snapshot)` accepts trusted operator journal records only and never discovers ownership by name alone.

`CloudBudgetSession(private_dir, approval, transport, guard_deployment_configured=True)` supplies the required hooks; `BudgetApproval.from_proposal` compares the independently trusted owner scope and `ApprovedComfySelection`. Reuse one journal path for this approval across restarts. The exclusive, private, atomic journal reserves each attempt before POST, retains worst-case reservations while invoices are unknown, and distinguishes estimates from authoritative final billing; it is no account-wide hard spend cap.

M3.4 must install the exact standalone `worker_guard.py`, verify its source/PID/own-Pod GET, then call `arm_guard(key, receipt)` before returning a ready probe. Heartbeat both the local session and the worker using that worker's timestamp during setup/download/inference; neither extends the absolute lease. `remaining_seconds` and local/remote supervision reserve the final 60s for cleanup. The worker reads only its existing Pod-scoped environment/PID1 credential and requests own-ID STOP; coordinator recovery separately verifies DELETE/GET404. Missing, duplicate or incomplete discovery remains uncertain and blocks new spend; unavailable worker startup, provider failures and stopped-disk charges remain recovery risks.

`cloud-smoke` composes those boundaries with bounded private SSH, pinned core-only ComfyUI setup and verified worker-only weight downloads; the account key stays on the laptop. It checks private prepared/managed artifacts against the same shot and decoded video contract, retaining each checksum while excluding volatile job/time/cost fields from parity; identical video bytes are not required.

The failure case withdraws remote heartbeats while local supervision continues, requires owned `EXITED`/absent-runtime readback before claiming independent STOP, then deletes and verifies GET404 even if STOP fails. Source/PID/own-Pod GET only attests current arming; it does not prove STOP permission or future completion. `cloud-recover` reuses the locked journal's original intents/spec without a new acquisition; neither command provides automatic generation resumption.
