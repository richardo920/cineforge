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

`execute_comfy(..., absolute_deadline=cap, clock=clock)` accepts a trusted same-clock cap that includes preflight and survives first-party transport admission, header/body sends, reads and final persistence; omitted caps preserve the legacy relative API. `execute_comfy` reserves `result_timeout_seconds` inside the unchanged absolute `timeout_seconds` cap (default reserve: `min(request_timeout_seconds, timeout_seconds / 2)`); setup/inference ends before that reserve. A confirmed output has one fixed transfer window, retries cannot renew it, and its validated server descriptor is checkpointed before download; cancellation also stays inside the original cap and an expired cap permits no new I/O, preserving the caller's lease cleanup reserve.

Selection hashes bind declarations, not owner identity; worker records are attestations, not remote inspection. Client bounds and best-effort job cancellation do not enforce cloud rental costs or guarantee worker shutdown.

Pod and Serverless will be selectable execution adapters sharing domain/model/workflow/artifact contracts. M2 implements private ComfyUI access to a manually prepared Pod; Serverless job APIs and cloud lifecycle automation remain future work.

## M3 provider review

[The owner-approved Runpod Pod scope](../../config/cloud/runpod-pod-proposal.json) compares Lambda and Modal and records a separate USD3 aggregate M3 validation cap with its [approval receipt](https://github.com/richardo920/cineforge/issues/48#issuecomment-6554788309). Live readiness remains false until M3.2/M3.3 safety gates, external credentials, explicit opt-in and fresh worker preflight pass; persistent storage, image publication and other GPU SKUs remain outside this scope.

`RunpodPodProvisioner(spec, transport, probe=..., before_create=..., event_sink=...)` reserves an `AcquisitionIntent` before POST, records each `LeaseSnapshot`, and returns a `GPUDevice` only after a separately supplied runtime probe; Runpod's `gpu.memory` is host RAM. Release verifies the exact journal-owned ID/name/session/lease independently of execution configuration, attempts deletion, and confirms fresh GET404; uncertain creation blocks replay, and unconfirmed cleanup retains the owned ID.

The required policy hook supplies fresh quote/approval/budget checks; the event sink persists private intent/state for M3.3 supervision. `RunpodHTTPTransport(api_key, allow_live=True)` only supplies bounded fixed-origin RESTv2 HTTPS: it does not authenticate owner approval, enforce a rental deadline, provision storage, install models, or execute workflows; a price race can still allocate before read-back rejects it. `restore_owned(intent, snapshot)` accepts trusted operator journal records only and never discovers ownership by name alone.

`CloudBudgetSession(private_dir, approval, transport, guard_deployment_configured=True)` supplies the required hooks; `BudgetApproval.from_proposal` compares the independently trusted owner scope and `ApprovedComfySelection`. Reuse one journal path for this approval across restarts. The exclusive, private, atomic journal reserves each attempt before POST, retains worst-case reservations while invoices are unknown, and distinguishes estimates from authoritative final billing; it is no account-wide hard spend cap.

M3.4 must install the exact standalone `worker_guard.py`, verify its source/PID/own-Pod GET, then call `arm_guard(key, receipt)` before returning a ready probe. Heartbeat both the local session and the worker using that worker's timestamp during setup/download/inference; neither extends the absolute lease. `remaining_seconds` and local/remote supervision reserve the final 60s for cleanup. The worker reads only its existing Pod-scoped environment/PID1 credential; owned STOP/EXITED polling shares bounded attempt/retry deadlines, skips repeat STOP while stopping, and cannot prove deletion or zero disk cost. Changed guard source requires fresh verified arming before READY; cached work rechecks the same source/deadline gate, and refused READY never populates its execution cache. Historical receipts remain available for owned cleanup. Coordinator recovery separately verifies DELETE/GET404. Missing, duplicate or incomplete discovery remains uncertain and blocks new spend; unavailable worker startup, provider failures and stopped-disk charges remain recovery risks.

`cloud-smoke` composes those boundaries with bounded private SSH, pinned core-only ComfyUI setup and verified worker-only weight downloads; the account key stays on the laptop. TCP readiness requires the owned private OpenSSH master's forward acknowledgement plus bounded ComfyUI health; plain loopback connectivity is insufficient. It checks private prepared/managed artifacts against the same shot and decoded video contract, retaining each checksum while excluding volatile job/time/cost fields from parity; identical video bytes are not required.

Mux commands deny direct fallback with `ProxyCommand=/usr/bin/false`. Tunnel teardown kills/reaps owned processes but retains at most eight private control namespaces per driver; `retained_control_namespace_paths` records residual paths for operator cleanup, without claiming current path ownership or deleting possibly replaced entries.

The failure case withdraws remote heartbeats while local supervision continues, requires owned `EXITED` lifecycle readback, then deletes and verifies GET404 even if STOP fails. Runtime presence is recorded separately: [the pinned official CLI classifier](https://github.com/runpod/runpodctl/blob/5b85fc45b7c850c776adc647137dab3dc0fdffe0/internal/podstate/podstate.go) treats `EXITED` as stopped even when telemetry remains reported. Source/PID/own-Pod GET only attests arming; `cloud-recover` releases journal-owned workers without acquiring or resuming generation.

## A20 node retention

Set `RunpodPodSpec(..., retain_node=True)` to skip A20's final deletion on success **and failure**; this flag is **A20-only** (`cloud_smoke` and `conditioning_run` still release). The default `False` preserves verified release. Successful retained runs remain `pending_human_review` with `release_verified=False`; failed runs raise `A20_INCOMPLETE_RETAIN_PRIVATE_RUN` and persist details in `run.json`.

Retention extends the journal lease and re-arms the on-pod guard before worker shutdown. Re-arming validates the controller clock and the own-Pod credential against the new plan before replacing any on-pod guard state, so a failed re-arm leaves the old guard live and still refreshed by supervision. The guard stops compute at **expiry − 60s** or after `idle_seconds` without a beat (default 120s). A detached local SSH child beats every ~10s, retries failures for up to 60s, then issues bounded retries of the on-pod stop on sustained failure and at expiry until dispatch reports stopped or the stop budget is exhausted, using only the pod's own credential and own-pod verification.

`retain_window_seconds` defaults to **3600s (1h)**; invalid/non-positive values are rejected. Expiry is `min(reserved_at + 14400, run_end + retain_window_seconds)`, so the effective window can be shorter than requested; windows of 90s or less are refused before mutation. Extension is one-shot, cannot revive an expired guard, and reserves additional cost against the approval aggregate; a window the aggregate cannot fully fund is clamped to the longest window it can fund, with the requested and effective windows both recorded, and only a window that cannot admit a bounded handoff at all is refused. The compute ceiling is the spec's `max_hourly_usd` plus the $0.02/hr reservation margin: at $1.25/hr the $5 aggregate admits at most ~12,370s (~3.4h) of retention past the original deadline, so a full 4h request is clamped to that instead of refused; disk billing continues until deletion.

Both bounds depend on a live guard or SSH reachability from the operator laptop. If the guard dies and the laptop cannot reach the pod, manual console termination is required to stop billing. Failed handoffs report unknown expiry/cost, preserve known guard bounds, and probe the old guard when liveness was never established: a live probe records the old guard bound and releases nothing; a dead or interrupted probe releases the pod through the provider API, skipping release only if the pod is already released. The journal remains non-terminal until #158 reconciliation.

`retained_node` preserves pod/region/GPU, snapshot state/rate, cached SSH address, driver version/mounts and the new guard plan. It adds acquisition time, requested/effective window, the 14400s bound, matching retention/lease expiry, detached heartbeat PID/state, and `max_retention_cost_usd` (snapshot hourly rate, or spec ceiling, × effective window hours). A snapshot/address is not proof of continued availability; a failed handoff marks the run incomplete. During execution, `worker.check()` can still enforce limits and release the pod.

To release: open the Runpod console → **Pods** → select the exact `retained_node.pod_id` from `run.json` → **Terminate** → confirm deletion. Verify that the pod disappears; **Stop** alone leaves disk billing. Console deletion does not reconcile the M5 journal: the retained lease stays non-terminal and `before_create` refuses new acquisitions while any journal record is non-terminal. Reuse the existing journal; do not bypass it to acquire again. M5 recovery/reconciliation is tracked in [#158](https://github.com/richardo920/cineforge/issues/158); `cloud-recover` currently covers M3 only, and future `restore_owned` recovery requires the original spec with `retain_node=True` to pass its spec-digest check.

Capabilities contain no GPU driver version, so `retained_node.driver_version` is null; available file-row `driver_mount` receipts are preserved in `driver_mounts`. No private key material is recorded; SSH information is null when no validated connection is cached.
