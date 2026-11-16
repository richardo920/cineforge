"""Fault-injected invariant failures must survive optimized Python execution."""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from cineforge import audio_process, native_remote_bootstrap, native_remote_worker, worker_bootstrap
from cineforge.cloud_budget import ConditioningBudgetApproval
from cineforge.comfy_model_precision import ModelGuard
from cineforge.comfy_runtime import ClosedComfyRuntime
from cineforge.conditioning import ConditioningSelection, VerifiedConditioning
from cineforge.native_remote import RemoteNativeSpeechRuntime
from cineforge.native_speech_integrity import NativeSpeechError
from cineforge.s2v import S2VServiceContext
from cineforge.ssh_worker import SSHWorker, SupervisedComfyTransport, WorkerError


class RuntimeInvariantTests(unittest.TestCase):
    """Bypass prior admission deliberately to exercise each last-line invariant."""

    def test_missing_process_pipes_fail_and_reap(self):
        for module, invoke, error in (
            (audio_process, lambda: audio_process.run_audio_process(
                ('/synthetic/tool',), payload=b'', timeout_seconds=1,
                max_output_bytes=4096), audio_process.AudioProcessError),
            (worker_bootstrap, lambda: worker_bootstrap.bounded_process(
                ['/synthetic/tool'], timeout=1), AssertionError),
        ):
            for missing in ('stdin', 'stdout'):
                with self.subTest(module=module.__name__, missing=missing):
                    process = Mock(stdin=Mock(), stdout=Mock(), pid=123)
                    setattr(process, missing, None)
                    with patch.object(module.subprocess, 'Popen', return_value=process), \
                            patch.object(module.os, 'killpg') as kill:
                        with self.assertRaises(error):
                            invoke()
                    kill.assert_called_once()
                    process.wait.assert_called_once_with(timeout=2)
                    getattr(process, 'stdout' if missing == 'stdin' else 'stdin').close.assert_called_once()

    def test_budget_requires_typed_approval(self):
        with self.assertRaises(AssertionError):
            ConditioningBudgetApproval.journal_directory.fget(SimpleNamespace(approval=object()))

    def test_native_dispatch_requires_typed_base_request(self):
        selected = SimpleNamespace(hardware=SimpleNamespace(
            gpu_name='synthetic', memory_mib=1, capability=(0, 0)))
        header = dict(operation='base', request={}, seed=1)
        with patch.object(native_remote_worker, '_selected', return_value=selected), \
                patch.object(native_remote_worker, 'require_s2v_hardware'), \
                patch.object(native_remote_worker, 'nr_token'), \
                patch.object(native_remote_worker, 'nr_read', return_value=b''), \
                patch.object(native_remote_worker, 'nr_write'), \
                patch.object(native_remote_worker, 'unpack', return_value=(header, b'')), \
                patch.object(native_remote_worker, 'NativeSpeechRuntime') as runtime, \
                patch.object(native_remote_worker.QwenBaseRequest, 'from_dict',
                             return_value=SimpleNamespace(language='English')):
            with self.assertRaises(AssertionError):
                native_remote_worker.remote_operation(
                    dict(operation='native_invoke', token='synthetic', profile={'id': 'synthetic'}),
                    Path('/synthetic'), time.monotonic() + 10, {})
            runtime.return_value.synthesize.assert_not_called()

    def test_remote_call_requires_deadline_and_binding_before_rpc(self):
        for deadline, binding in ((None, {}), (time.monotonic() + 10, None)):
            bridge = SimpleNamespace(_bound_worker=Mock(), _worker=Mock(),
                                     _deadline=deadline, _binding=binding)
            with self.subTest(deadline=deadline, binding=binding):
                with self.assertRaises(NativeSpeechError):
                    RemoteNativeSpeechRuntime._call(bridge, 'native_preflight')
                bridge._worker.native_request.assert_not_called()
                self.assertTrue(bridge._unresolved)

    def test_remote_bound_worker_requires_service_context(self):
        bridge = SimpleNamespace(_unresolved=False, _deadline=1, _binding={},
                                 _worker=SimpleNamespace(selection=object()))
        with self.assertRaises(AssertionError):
            RemoteNativeSpeechRuntime._bound_worker(bridge)

    def test_remote_initialize_requires_saved_deadline(self):
        bridge = SimpleNamespace(_worker=Mock(), _bound_worker=Mock(), _deadline=None,
                                 _binding={'package': hashlib.sha256(b'').hexdigest()})
        with self.assertRaises(AssertionError):
            RemoteNativeSpeechRuntime._initialize(bridge, b'', time.monotonic() + 10)

    def test_remote_initialize_requires_plan_and_service_context(self):
        for plan in (None, SimpleNamespace(pod_id='synthetic', session_id='synthetic')):
            worker = SimpleNamespace(native_admission=Mock(), plan=plan, selection=object(),
                                     budget=SimpleNamespace(remaining_seconds=lambda _: 900), _key='key')
            bridge = SimpleNamespace(_worker=worker, _binding=None)
            with self.subTest(plan=plan):
                with self.assertRaises(AssertionError):
                    RemoteNativeSpeechRuntime._initialize(bridge, b'', time.monotonic() + 10)
                self.assertIsNone(bridge._binding)

    def test_remote_measurement_requires_deadline_before_publishing(self):
        bridge = SimpleNamespace(
            _timeout=10, _parent_deadline=None, _lock=threading.Lock(),
            _capabilities_attempted=False, _staging_attempted=False, _unresolved=False,
            _initialize=Mock(), _call=Mock(return_value={}), _validate_capabilities=Mock(),
            _worker=Mock(), _bound_worker=Mock(), _deadline=None,
            _capabilities_sha256=None, _capabilities_payload=None)
        with patch('cineforge.native_remote.native_package', return_value=(b'', {})):
            with self.assertRaises(NativeSpeechError):
                RemoteNativeSpeechRuntime.measure_capabilities(bridge)
        self.assertTrue(bridge._unresolved)
        self.assertIsNone(bridge._capabilities_payload)
        self.assertFalse(bridge._lock.locked())

    def test_native_preflight_requires_runtime_requirements(self):
        with patch.object(native_remote_bootstrap, 's2v_worker',
                          return_value=SimpleNamespace(reference_runtime=None)):
            with self.assertRaises(AssertionError):
                native_remote_bootstrap.nr_preflight(time.monotonic() + 10)

    def test_mix_requires_both_measurements_before_gain(self):
        from test_audio_mix import IDENTITY, Meter, scene

        from cineforge import audio_mix

        args = scene()
        for loudness, peak in ((None, -10), (-23, None)):
            before = SimpleNamespace(issues=(), tool=IDENTITY, measurement_status='measured',
                                     integrated_lufs=SimpleNamespace(value=loudness),
                                     true_peak_dbtp=SimpleNamespace(value=peak))
            with self.subTest(loudness=loudness, peak=peak):
                with patch.object(audio_mix, 'analyze_wav', return_value=before) as analyze:
                    with self.assertRaisesRegex(audio_mix.MixError, '^soundtrack mix rejected$'):
                        audio_mix.mix_scene(**args, preset=audio_mix.INTERNAL_REVIEW_STEREO,
                                            output_id='mix', output_revision=1,
                                            tool=Meter(), expected_tool=IDENTITY)
                self.assertEqual(analyze.call_count, 1)

    def test_precision_cleanup_requires_nonce_before_receipt(self):
        guard = SimpleNamespace(_current=lambda: {'nonce': None}, _context=Mock(),
                                _validate=Mock(), nonce=None, step=1, inside=False,
                                _disarm=Mock(), _time=Mock(), _receipt=Mock())
        with self.assertRaises(AssertionError):
            ModelGuard._on_cleanup(guard, object())
        guard._disarm.assert_called_once()
        guard._receipt.assert_not_called()

    def test_precision_receipt_requires_request(self):
        with self.assertRaises(AssertionError):
            ModelGuard._receipt(SimpleNamespace(request=None), object())

    def test_supervisor_requires_plan_before_heartbeat(self):
        worker = SimpleNamespace(_stop=Mock(), check=Mock(), plan=None, _failed=Mock(),
                                 _close_tunnel=Mock(), _request=Mock())
        worker._stop.wait.side_effect = (False, True)
        SSHWorker._supervise(worker)
        worker._failed.set.assert_called_once()
        worker._close_tunnel.assert_called_once()
        worker._request.assert_not_called()

    def test_worker_prepare_requires_deadline_and_selection(self):
        service = object.__new__(S2VServiceContext)
        runtime = object.__new__(ClosedComfyRuntime)
        for deadline in (None, time.monotonic() + 10):
            worker = SimpleNamespace(selection=service, check=Mock(), plan=object(),
                budget=SimpleNamespace(remaining_seconds=lambda _: 900), _key='key',
                _s2v_setup_attempted=False, _require_s2v=Mock(), _prepare_closed_comfy=Mock())
            with self.subTest(deadline=deadline):
                with patch('cineforge.ssh_worker.request_deadline', return_value=deadline):
                    with self.assertRaises(AssertionError):
                        SSHWorker.prepare(worker, None, runtime=runtime)
                worker._prepare_closed_comfy.assert_not_called()

    def test_closed_worker_prepare_requires_plan_before_accepting_verification(self):
        worker = SimpleNamespace(selection=object.__new__(S2VServiceContext),
                                 plan=None, _runtime_evidence=Mock(return_value={}))
        runtime = Mock()
        runtime.prepare.return_value = {'runtime': {}, 'verification': {}}
        with patch.object(S2VServiceContext, 'worker_profile', new_callable=PropertyMock,
                          return_value={'id': 'synthetic'}), \
                patch('cineforge.ssh_worker.worker_bootstrap.validate_s2v_runtime'), \
                patch('cineforge.ssh_worker.WorkerVerification.from_dict'):
            with self.assertRaises(WorkerError):
                SSHWorker._prepare_closed_comfy(worker, object(), runtime, time.monotonic() + 10)
        runtime._poison.assert_called_once()
        self.assertFalse(hasattr(worker, '_prepared_s2v'))

    def test_native_request_requires_plan_before_dispatch(self):
        worker = SimpleNamespace(native_admission=Mock(), plan=None, _request=Mock(),
            budget=SimpleNamespace(remaining_seconds=lambda _: 900), _key='key')
        with self.assertRaises(AssertionError):
            SSHWorker.native_request(worker, {'operation': 'native_preflight'},
                                     absolute_deadline=time.monotonic() + 10)
        worker._request.assert_not_called()

    def test_prompt_requires_closed_runtime_before_transport(self):
        worker = SimpleNamespace(check=Mock(), selection=object.__new__(S2VServiceContext),
                                 verify_comfy_current=Mock(), _closed_comfy=None)
        transport = Mock()
        with self.assertRaises(AssertionError):
            SupervisedComfyTransport(worker, transport).request(
                'POST', '/prompt', {}, timeout_seconds=1, max_bytes=4096)
        transport.request.assert_not_called()

    def test_conditioning_trace_requires_current_character(self):
        shot = object()
        gate = SimpleNamespace(require=Mock(), workflow_digest='synthetic',
                               bindings=(object(),), requests=(shot,), character=object(),
                               registry=SimpleNamespace(lookup=lambda _: None))
        with patch('cineforge.conditioning.ExecutionRequest'), \
                patch('cineforge.conditioning.profile.model'):
            with self.assertRaises(AssertionError):
                VerifiedConditioning.trace(gate, shot)

    def test_conditioning_worker_requires_staging(self):
        selection = SimpleNamespace(gate=SimpleNamespace(
            require_submission=Mock(), requests=(object(),), staging=None), request=Mock())
        worker = SimpleNamespace(record={'worker_id': 'synthetic'})
        with self.assertRaises(AssertionError):
            ConditioningSelection.require_execution_worker(selection, worker)

    def test_comfy_call_requires_deadline_and_expiry_before_rpc(self):
        for deadline, expiry in ((None, 1), (1, None)):
            bridge = SimpleNamespace(_deadline=deadline, _call=Mock())
            runtime = SimpleNamespace(_bridge=bridge, _expires_unix=expiry)
            with self.subTest(deadline=deadline, expiry=expiry):
                with self.assertRaises(AssertionError):
                    ClosedComfyRuntime._call(runtime, 'synthetic', 1)
                bridge._call.assert_not_called()

    def test_comfy_prepare_requires_bridge_deadline_and_expiry(self):
        for deadline, expiry in ((None, 1), (1, None)):
            bridge = SimpleNamespace(_deadline=deadline, _expires_unix=expiry, _upload=Mock())
            runtime = SimpleNamespace(_attempted=False, _admit=Mock(), _bridge=bridge)
            with self.subTest(deadline=deadline, expiry=expiry):
                with self.assertRaises(AssertionError):
                    ClosedComfyRuntime.prepare(runtime, Mock(), object(), deadline=1)
                bridge._upload.assert_not_called()
                self.assertTrue(runtime._attempted)

    def test_comfy_precision_requires_binding_before_rpc(self):
        runtime = SimpleNamespace(_prepared=True, _precision=None, _precision_used=set(),
            _admit=Mock(), _bridge=SimpleNamespace(_binding=None), _call=Mock())
        selection = SimpleNamespace(gate=SimpleNamespace(utterance=SimpleNamespace(language='en')))
        with patch('cineforge.native_remote_package.native_package', return_value=(b'', {})):
            with self.assertRaises(AssertionError):
                ClosedComfyRuntime.begin_precision(runtime, Mock(), selection, deadline=1)
        runtime._call.assert_not_called()


class OptimizedRuntimeTests(unittest.TestCase):
    def test_invariant_failures_match_with_optimization(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env.pop('PYTHONOPTIMIZE', None)
        env['PYTHONPATH'] = os.pathsep.join((str(root / 'src'), str(root / 'tests')))
        results = []
        for flags in ([], ['-O']):
            result = subprocess.run(
                [sys.executable, *flags, '-m', 'unittest',
                 'test_runtime_checks.RuntimeInvariantTests', '-v'],
                cwd=root, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Ignore timing; require identical case names and outcomes in both interpreters.
            results.append([line for line in result.stderr.splitlines() if line.startswith('test_')])
        self.assertEqual(results[0], results[1])
        expected = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeInvariantTests)
        self.assertEqual(len(results[0]), expected.countTestCases())

    def test_src_has_no_bare_asserts(self):
        root = Path(__file__).resolve().parents[1] / 'src'
        remaining = []
        for path in root.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                if isinstance(node, ast.Assert):
                    remaining.append(f'{path.relative_to(root)}:{node.lineno}')
        self.assertEqual(remaining, [])
