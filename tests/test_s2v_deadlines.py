"""Deterministic controller clocks and first-party RPC/provider paths; no SSH or GPU."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_s2v_run as run_fixtures
import test_s2v_stage as stage_fixtures
import test_ssh_tunnel as tunnel_fixtures
from test_s2v import context
from test_s2v_profile import png

from cineforge.private_store import PrivateRunDirectory
from cineforge.s2v import S2VError, select_s2v
from cineforge.s2v_service import S2VServiceContext
from cineforge.ssh_worker import SSHWorker, WorkerError
from cineforge.workflows import canonical


class PairedDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.f = stage_fixtures.PairTransferTests('test_complete_pair_exact_bytes_metadata_and_consumed_repeated_request')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.driver, self.selection = self.f.driver, self.f.selection
        self.driver._host, self.driver._port = "127.0.0.1", 22
        self.now = 99.0
        self.clock = patch('cineforge.ssh_worker.time.monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_entry_expiry_and_invalid_caps_make_no_admission_or_wire_calls(self):
        with patch.object(self.driver, '_require_s2v') as admission:
            for cap in (99, 98, True, float('nan'), float('inf')):
                with self.subTest(cap=cap), self.assertRaises(WorkerError):
                    self.driver.stage_s2v(self.selection, absolute_deadline=cap)
            admission.assert_not_called()
        self.assertEqual(self.f.requests, [])
        self.assertFalse(self.driver._s2v_consumed)

    def test_lock_wait_counts_against_original_parent_cap(self):
        lock = self.driver._s2v_lock
        owner = self
        class DelayedLock:
            def __enter__(self):
                lock.acquire()
                owner.now = 101
            def __exit__(self, *args):
                lock.release()
        self.driver._s2v_lock = DelayedLock()
        with self.assertRaises(WorkerError):
            self.driver.stage_s2v(self.selection, absolute_deadline=100)
        self.assertFalse(lock.locked())
        self.assertEqual(self.f.requests, [])
        self.assertFalse(self.driver._s2v_consumed)

    def test_real_request_preflight_crossing_parent_prevents_process_dispatch(self):
        original = S2VServiceContext.preflight
        inside, dispatched = [False], []
        def preflight(service):
            original(service)
            if inside[0]:
                self.now = 101
        def process(*args, **kwargs):
            dispatched.append((self.now, kwargs.get('absolute_deadline')))
            return b'{}'
        def request(body, timeout, **kwargs):
            inside[0] = True
            try:
                return SSHWorker._request(self.driver, body, timeout, **kwargs)
            finally:
                inside[0] = False
        self.driver._request = request
        with patch.object(S2VServiceContext, 'preflight', preflight), \
                patch('cineforge.ssh_worker.run_process', side_effect=process):
            # Exact old relative-only request path dispatches after preflight.
            inside[0] = True
            SSHWorker._request(self.driver, {'operation': 's2v_pair_begin'}, 1)
            inside[0] = False
            self.assertEqual(dispatched, [(101, None)])
            self.now = 99
            dispatched.clear()
            with self.assertRaises(WorkerError):
                self.driver.stage_s2v(self.selection, absolute_deadline=100)
        self.assertEqual(dispatched, [])
        self.assertEqual(len(self.driver._s2v_consumed), 1)

    def test_all_chunks_forward_one_stricter_parent_through_real_request(self):
        root = self.f.root / 'large-deadline'
        root.mkdir()
        gate, _ = context(root, actor_png=png(compressed=False))
        gate = stage_fixtures.reapprove(gate, service=self.f.gate.service)
        self.selection = select_s2v(gate)
        sent = []
        def process(*args, **kwargs):
            body = json.loads(kwargs['payload'])
            sent.append((body['operation'], kwargs['absolute_deadline'], kwargs['timeout']))
            result = self.f.rpc(body, kwargs['timeout'])
            self.now += .01
            return canonical(result)
        self.driver._request = lambda body, timeout, **kw: SSHWorker._request(self.driver, body, timeout, **kw)
        with patch('cineforge.ssh_worker.run_process', side_effect=process):
            proof = self.driver.stage_s2v(self.selection, absolute_deadline=105)
        self.assertEqual(proof.worker_id, 'owned-worker')
        self.assertGreater(sum(op == 's2v_pair_chunk' for op, _, _ in sent), 2)
        self.assertEqual({cap for _, cap, _ in sent}, {105})
        self.assertTrue(all(0 < timeout <= 6 for _, _, timeout in sent))
        self.assertEqual((sent[0][0], sent[-1][0]), ('s2v_pair_begin', 's2v_pair_finish'))
        self.assertTrue(all('absolute_deadline' not in body for body in self.f.requests))

    def test_longer_parent_does_not_extend_phase_cap(self):
        sent = []
        rpc = self.driver._request
        def request(body, timeout, **kw):
            sent.append(kw['absolute_deadline'])
            return rpc(body, timeout, **kw)
        self.driver._request = request
        self.driver.stage_s2v(self.selection, absolute_deadline=1000)
        self.assertEqual(set(sent), {189})

    def test_lease_sample_before_delayed_budget_callback_never_renews_reserve(self):
        def remaining(_):
            self.now = 103
            return 123
        self.driver.budget.remaining_seconds = remaining
        with self.assertRaises(WorkerError):
            self.driver.stage_s2v(self.selection)
        self.assertEqual(self.f.requests, [])

    def test_late_chunk_or_finish_keeps_consumed_pair_and_never_replays(self):
        for operation in ('s2v_pair_chunk', 's2v_pair_finish'):
            with self.subTest(operation=operation):
                self.f.remote = self.f.root / operation
                (self.f.remote / 'ComfyUI/input').mkdir(parents=True, mode=0o700)
                driver = self.f.new_driver()
                def request(body, timeout, **kwargs):
                    reply = self.f.rpc(body, timeout, **kwargs)
                    if body['operation'] == operation:
                        self.now = 101
                    return reply
                driver._request = request
                self.now = 99
                with self.assertRaises(WorkerError):
                    driver.stage_s2v(self.selection, absolute_deadline=100)
                sent = len(self.f.requests)
                self.now = 99
                with self.assertRaises(WorkerError):
                    driver.stage_s2v(self.selection, absolute_deadline=105)
                self.assertEqual(len(self.f.requests), sent)
                self.assertTrue(driver._s2v_consumed)

    def test_real_lock_release_and_late_final_proof_cannot_return_staged_proof(self):
        lock = self.driver._s2v_lock
        owner = self
        class LateRelease:
            def __enter__(self):
                lock.acquire()
            def __exit__(self, *args):
                lock.release()
                owner.now = 101
        self.driver._s2v_lock = LateRelease()
        with self.assertRaises(WorkerError):
            self.driver.stage_s2v(self.selection, absolute_deadline=100)
        self.assertFalse(lock.locked())
        self.assertEqual(self.f.requests[-1]['operation'], 's2v_pair_finish')
        self.assertTrue(self.driver._s2v_consumed)
        path = self.f.remote / 'ComfyUI/input' / self.f.gate.media['audio']['basename']
        self.assertEqual(path.read_bytes(), self.f.gate.speech.payload)


class ExecutionDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.f = run_fixtures.S2VRunTests('test_success_pins_actual_fixture_bytes_and_all_lineage_without_human_authority')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.now = 100.0

    def run_scene(self, **kw):
        return self.f.run_scene(clock=lambda: self.now, **kw)

    def test_expired_or_invalid_parent_refuses_before_callback_directory_or_calls(self):
        with patch.object(type(self.f.selection.gate), 'preflight') as admission:
            for cap in (100, 99, True, float('nan'), float('inf')):
                with self.subTest(cap=cap), self.assertRaises(S2VError):
                    self.run_scene(absolute_deadline=cap)
            admission.assert_not_called()
        self.assertFalse(self.f.output.exists())
        self.assertEqual(self.f.wire.calls, [])

    def test_foreign_preflight_crossing_stricter_parent_makes_no_calls(self):
        gate = self.f.selection.gate
        original = gate.verify_speech
        def proof(*args):
            self.now = 106
            return original(*args)
        self.f.selection = replace(self.f.selection, gate=replace(gate, verify_speech=proof))
        with self.assertRaises(S2VError):
            self.run_scene(absolute_deadline=105)
        self.assertEqual(self.f.wire.calls, [])
        self.assertFalse(self.f.output.exists())

    def test_stricter_parent_is_recorded_and_forwarded_without_renewal(self):
        from cineforge.comfy_run import execute_comfy
        caps = []
        def execute(*args, **kwargs):
            caps.append(kwargs['absolute_deadline'])
            return execute_comfy(*args, **kwargs)
        result = self.run_scene(absolute_deadline=105, execute=execute)
        self.assertEqual(caps, [105])
        self.assertEqual(result['controller_deadline'], 105)
        self.assertTrue(all(0 < c[3] <= 5 for c in self.f.wire.calls))
        durable = json.loads((self.f.output / 'run.json').read_text())
        self.assertEqual((durable['outcome'], durable['human_review']), ('generated', 'unavailable'))

    def test_delayed_initial_budget_sample_does_not_shift_lease_cap(self):
        def remaining(_):
            self.now = 161
            return 180
        self.f.driver.budget.remaining_seconds = remaining
        with self.assertRaises(S2VError):
            self.run_scene(absolute_deadline=900)
        self.assertEqual(self.f.wire.calls, [])
        self.assertFalse(self.f.output.exists())

    def test_late_inspection_and_real_directory_close_preserve_identity_not_readiness(self):
        original = PrivateRunDirectory.__exit__
        def close(store, *args):
            original(store, *args)
            if store.path == self.f.output:
                self.now = 106
        with patch.object(PrivateRunDirectory, '__exit__', close), self.assertRaises(S2VError):
            self.run_scene(absolute_deadline=105)
        durable = json.loads((self.f.output / 'run.json').read_text())
        self.assertEqual((durable['outcome'], durable['human_review']), ('generated', 'unavailable'))
        self.assertEqual(durable['manifest']['shot_runs'][0]['status'], 'succeeded')
        self.assertNotIn('technical_result', durable)
        self.assertEqual(sum(c[1] == '/prompt' for c in self.f.wire.calls), 1)


class EndpointDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.f = tunnel_fixtures.TunnelTests('test_owned_forward_ack_precedes_health_and_endpoint_is_cached')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def test_real_endpoint_delayed_guard_cannot_launch_child_past_parent(self):
        with self.f.harness() as wire:
            original = self.f.worker.check
            def delayed():
                original()
                wire.now = 2
            self.f.worker.check = delayed
            with self.assertRaises(WorkerError):
                self.f.worker.endpoint(absolute_deadline=1)
            self.assertEqual(wire.processes, [])
            self.assertEqual(wire.events, [])
            self.assertEqual(self.f.worker._endpoint, '')

    def test_real_endpoint_command_preflight_cannot_launch_child_past_parent(self):
        with self.f.harness() as wire:
            original = self.f.worker._command
            def delayed():
                command = original()
                wire.now = 2
                return command
            self.f.worker._command = delayed
            with self.assertRaises(WorkerError):
                self.f.worker.endpoint(absolute_deadline=1)
            self.assertEqual(wire.processes, [])
            self.assertEqual(self.f.worker._endpoint, '')

    def test_mux_processes_receive_parent_cap_and_final_lock_release_is_checked(self):
        worker = self.f.worker
        lock = worker._endpoint_lock
        with self.f.harness() as wire:
            sent = []
            def mux(command, **options):
                sent.append(options['absolute_deadline'])
                return wire.mux(command, **options)
            class LateRelease:
                def __enter__(self):
                    lock.acquire()
                def __exit__(self, *args):
                    lock.release()
                    wire.now = 2
            worker._endpoint_lock = LateRelease()
            with patch('cineforge.ssh_worker.run_process', side_effect=mux), self.assertRaises(WorkerError):
                worker.endpoint(absolute_deadline=1)
            self.assertEqual(sent, [1, 1])
            self.assertTrue(wire.processes)
            self.assertTrue(all(p.poll() is not None for p in wire.processes))
            self.assertEqual(worker._endpoint, '')
            self.assertIsNone(worker._tunnel)

    def test_cached_endpoint_guard_failure_cannot_return_stale_success(self):
        with self.f.harness() as wire:
            worker = self.f.worker
            worker.endpoint(absolute_deadline=1)
            original = worker.check
            count = []
            def failing():
                original()
                count.append(True)
                if len(count) == 2:
                    raise WorkerError('synthetic late guard refusal')
            worker.check = failing
            with self.assertRaises(WorkerError):
                worker.endpoint(absolute_deadline=1)
            self.assertEqual(worker._endpoint, '')
            self.assertTrue(all(p.poll() is not None for p in wire.processes))
