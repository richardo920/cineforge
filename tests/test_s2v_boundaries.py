"""Actual first-party provider boundaries over synthetic media and a fake wire only."""
from __future__ import annotations

import tempfile
import traceback
import unittest
from dataclasses import fields, replace
from pathlib import Path

from test_s2v import context, worker
from test_s2v_run import S2VWire

from cineforge import s2v_profile as profile
from cineforge.comfy import ComfyExecutionProvider
from cineforge.providers import ExecutionRequest, ProviderError
from cineforge.s2v import S2VError, StagedS2VPair, select_s2v
from cineforge.workflows import canonical


def forged(record, **changes):
    value = object.__new__(type(record))
    for item in fields(record):
        object.__setattr__(value, item.name, changes.get(item.name, getattr(record, item.name)))
    return value


def provider_for(gate):
    selection = select_s2v(gate)
    verification = worker(gate)
    proof = StagedS2VPair("owned-worker", gate.service.approval.subject_sha256,
                         selection.approved_subject_sha256, canonical(gate.media).decode())
    selection = selection.bind_worker(verification, proof)
    wire = S2VWire(selection)
    provider = ComfyExecutionProvider(wire, selection.workflow,
        verified_server_revision=verification.record["server_revision"],
        verified_worker_id="owned-worker")
    return provider, wire


class S2VCallbackBoundaryTests(unittest.TestCase):
    def test_every_actor_native_and_authorize_callback_window_rechecks_voice_before_post(self):
        # Five proof callbacks/two authorizations occur across translation and final
        # require_submission. The old implementation submitted after callback five.
        for name, count in (("verify_actor", 5), ("verify_speech", 5), ("authorize", 2)):
            for at in range(1, count + 1):
                with self.subTest(callback=name, at=at), tempfile.TemporaryDirectory() as temp:
                    gate, _ = context(Path(temp).resolve())
                    original = getattr(gate, name)
                    armed, calls = [False], [0]
                    def callback(*args):
                        value = original(*args)
                        if armed[0]:
                            calls[0] += 1
                            if calls[0] == at:
                                gate.ledger.revoke(gate.utterance.voice.permission.rights,
                                                   reason="CANARY callback revocation")
                        return value
                    gate = replace(gate, **{name: callback})
                    provider, wire = provider_for(gate)
                    armed[0] = True
                    with self.assertRaises((ProviderError, S2VError)):
                        provider.submit(ExecutionRequest(gate.shot, profile.model()), "one-attempt")
                    self.assertEqual(calls[0], at)
                    self.assertEqual(wire.calls, [])

    def test_proof_callback_replacement_of_original_bytes_or_metadata_cannot_post(self):
        for change in ("actor_bytes", "timebase", "actor_rights", "recording_rights"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                gate, _ = context(Path(temp).resolve())
                native, armed = gate.verify_speech, [False]
                def callback(*args):
                    value = native(*args)
                    if armed[0]:
                        if change == "actor_bytes":
                            gate.actor.path.write_bytes(b"CANARY replacement")
                        elif change == "timebase":
                            object.__setattr__(gate.sheet.timebase, "ticks_per_second", True)
                        else:
                            permission = (gate.actor.artifact.permission if change == "actor_rights"
                                          else gate.speech.request.recording_permission)
                            gate.ledger.revoke(permission.rights, reason="CANARY")
                    return value
                gate = replace(gate, verify_speech=callback)
                provider, wire = provider_for(gate)
                armed[0] = True
                with self.assertRaises((ProviderError, S2VError)):
                    provider.submit(ExecutionRequest(gate.shot, profile.model()), "one-attempt")
                self.assertEqual(wire.calls, [])

    def test_forged_typed_proof_and_nested_leaves_reject_before_comparison(self):
        for kind in ("character_bool_revision", "character_bool_schema", "evidence_bool_schema",
                     "evidence_uri_type", "actor_digest_type", "zero_native_receipt",
                     "native_worker_type", "native_request_type", "native_scope_type"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                gate, _ = context(Path(temp).resolve())
                actor, native = gate.verify_actor, gate.verify_speech
                def actor_callback(*args):
                    value = actor(*args)
                    if kind.startswith("character_"):
                        change = {"revision": True} if kind.endswith("revision") else {"schema_version": True}
                        return forged(value, character=forged(value.character, **change))
                    if kind.startswith("evidence_"):
                        change = {"uri": True} if kind.endswith("type") else {"schema_version": True}
                        return forged(value, evidence=forged(value.evidence, **change))
                    return forged(value, reference_sha256=True) if kind == "actor_digest_type" else value
                def native_callback(*args):
                    value = native(*args)
                    changes = {"zero_native_receipt": {"native_receipt_sha256": "0" * 64},
                        "native_worker_type": {"worker_id": True},
                        "native_request_type": {"request_sha256": True},
                        "native_scope_type": {"scope_sha256": True}}
                    return forged(value, **changes.get(kind, {}))
                bad = replace(gate, verify_actor=actor_callback, verify_speech=native_callback)
                with self.assertRaises(S2VError) as caught:
                    select_s2v(bad)
                self.assertIsNone(caught.exception.__context__)
                self.assertNotIn("CANARY", "".join(traceback.format_exception(caught.exception)))

    def test_later_native_callback_cannot_mutate_an_already_returned_actor_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            gate, _ = context(Path(temp).resolve())
            actor, native, retained = gate.verify_actor, gate.verify_speech, []
            def actor_callback(*args):
                value = actor(*args)
                value = replace(value, evidence=replace(value.evidence))
                retained.append(value)
                return value
            def native_callback(*args):
                value = native(*args)
                object.__setattr__(retained[-1].evidence, "schema_version", True)
                return value
            with self.assertRaises(S2VError):
                select_s2v(replace(gate, verify_actor=actor_callback, verify_speech=native_callback))


if __name__ == "__main__":
    unittest.main()
