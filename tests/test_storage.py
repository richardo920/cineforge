"""Synthetic durable-run acceptance and crash/fault probes; no live adapters."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cineforge.contracts import RunManifest, RunStatus
from cineforge.mock import MockPlan, execute_mock, load_mock_plan, mock_run_id
from cineforge.providers import FakeExecutionProvider, ProviderError
from cineforge.storage import LocalRunStore, StorageError

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic-plan.json'


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cineforge-m15-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / 'run'
        self.plan = load_mock_plan(EXAMPLE)

    def checkpoint(self):
        return json.loads((self.output / 'checkpoint.json').read_bytes())

    def rewrite(self, data):
        (self.output / 'checkpoint.json').write_text(json.dumps(data), encoding='utf-8')

    def assert_rejected(self, plan=None, **kwargs):
        with patch('cineforge.mock.FakeExecutionProvider') as provider:
            with self.assertRaises((StorageError, OSError, ValueError)):
                execute_mock(plan or self.plan, output_dir=self.output, resume=True, **kwargs)
            provider.assert_not_called()

    def test_actual_payloads_and_completed_resume_preserve_bytes_without_submission(self):
        result = execute_mock(self.plan, output_dir=self.output)
        stored = RunManifest.from_dict(self.checkpoint()['manifest'])
        self.assertEqual(result.manifest, stored)
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.output.iterdir()}
        for artifact in stored.artifacts:
            payload = (self.output / artifact.uri).read_bytes()
            self.assertEqual(artifact.size_bytes, len(payload))
            self.assertEqual(artifact.sha256, hashlib.sha256(payload).hexdigest())
            self.assertTrue(json.loads(payload)['synthetic'])
        with patch.object(FakeExecutionProvider, 'submit', side_effect=AssertionError('resubmit')):
            resumed = execute_mock(self.plan, output_dir=self.output, resume=True)
        self.assertEqual(result, resumed)
        self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                                  for path in self.output.iterdir()})

    def test_missing_and_corrupt_completed_payload_rejected_before_execution(self):
        for mode in ('missing', 'checksum', 'size'):
            with self.subTest(mode=mode):
                self.output = self.root / mode
                result = execute_mock(self.plan, output_dir=self.output)
                path = self.output / result.manifest.artifacts[-1].uri
                if mode == 'missing':
                    path.unlink()
                elif mode == 'checksum':
                    raw = path.read_bytes()
                    path.write_bytes(b'X' + raw[1:])
                else:
                    data = self.checkpoint()
                    data['manifest']['artifacts'][-1]['size_bytes'] += 1
                    self.rewrite(data)
                self.assert_rejected()

    def test_full_input_and_run_identity_binding(self):
        execute_mock(self.plan, run_id='fixed-run', output_dir=self.output)
        self.assert_rejected(run_id='other-run')
        for section, field, value in (
            ('models', 'license', 'LicenseRef-Changed'),
            ('models', 'commercial_use', True), ('devices', 'memory_mib', 2),
            ('plan', 'prompt', 'changed prompt'), ('plan', 'revision', 2),
        ):
            data = copy.deepcopy(self.plan.to_dict())
            if section == 'plan':
                data[section]['shots'][0][field] = value
            else:
                data[section][0][field] = value
            changed = MockPlan.from_json(json.dumps(data))
            self.assert_rejected(changed, run_id='fixed-run')
        data = self.plan.to_dict()
        data['models'].append(dict(data['models'][0], model_id='unused-model'))
        self.assert_rejected(MockPlan.from_json(json.dumps(data)), run_id='fixed-run')

    def test_invalid_records_rejected(self):
        execute_mock(self.plan, output_dir=self.output)
        valid = self.checkpoint()
        for field, value in (('schema_version', True), ('run_id', 'wrong'),
                             ('shot_revision', 2), ('idempotency_key', 'wrong'),
                             ('job_id', 'wrong'), ('attempt', 2),
                             ('uri', '../outside.json'), ('uri', '/outside.json')):
            data = copy.deepcopy(valid)
            if field == 'schema_version':
                data[field] = value
            elif field == 'run_id':
                data['manifest'][field] = value
            elif field == 'uri':
                data['manifest']['artifacts'][0][field] = value
            else:
                data['manifest']['shot_runs'][0][field] = value
            self.rewrite(data)
            self.assert_rejected()
        for raw in ('{', '{"schema_version":1,"schema_version":1}',
                    '{"schema_version":NaN}', '[' * 2000):
            (self.output / 'checkpoint.json').write_text(raw, encoding='utf-8')
            self.assert_rejected()

    def test_failed_job_is_durable_and_retries_once(self):
        provider = FakeExecutionProvider(fail=True)
        with patch('cineforge.mock.FakeExecutionProvider', return_value=provider):
            with self.assertRaises(ProviderError):
                execute_mock(self.plan, output_dir=self.output)
        failed = RunManifest.from_dict(self.checkpoint()['manifest'])
        self.assertEqual(len(failed.shot_runs), 1)
        self.assertEqual(failed.shot_runs[0].status, RunStatus.FAILED)
        self.assertTrue(failed.shot_runs[0].error)
        result = execute_mock(self.plan, output_dir=self.output, resume=True)
        self.assertEqual(result.manifest.shot_runs[0], failed.shot_runs[0])
        self.assertEqual([run.attempt for run in result.manifest.shot_runs], [1, 2, 1])
        self.assertEqual(len(result.manifest.artifacts), 2)

    def test_provider_exception_is_durable_and_running_interruption_replays(self):
        for failure in (RuntimeError('provider fault'), KeyboardInterrupt('interrupted')):
            with self.subTest(failure=type(failure).__name__):
                self.output = self.root / type(failure).__name__
                provider = FakeExecutionProvider()
                original = provider.status
                calls = 0

                def status(job_id):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise failure
                    return original(job_id)

                with patch('cineforge.mock.FakeExecutionProvider', return_value=provider), \
                        patch.object(provider, 'status', side_effect=status):
                    with self.assertRaises((ProviderError, KeyboardInterrupt)):
                        execute_mock(self.plan, output_dir=self.output)
                prior = RunManifest.from_dict(self.checkpoint()['manifest']).shot_runs[0]
                expected = (RunStatus.FAILED if isinstance(failure, Exception)
                            else RunStatus.RUNNING)
                self.assertEqual(prior.status, expected)
                result = execute_mock(self.plan, output_dir=self.output, resume=True)
                first_success = next(run for run in result.manifest.shot_runs
                                     if run.status == RunStatus.SUCCEEDED)
                self.assertEqual(first_success.attempt, 2 if expected == RunStatus.FAILED else 1)

    def interrupt_after_first_shot(self):
        provider = FakeExecutionProvider()
        original = provider.status
        calls = 0

        def status(job_id):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise KeyboardInterrupt('synthetic interruption')
            return original(job_id)

        with patch('cineforge.mock.FakeExecutionProvider', return_value=provider), \
                patch.object(provider, 'status', side_effect=status):
            with self.assertRaises(KeyboardInterrupt):
                execute_mock(self.plan, output_dir=self.output)
        manifest = RunManifest.from_dict(self.checkpoint()['manifest'])
        self.assertEqual([run.status for run in manifest.shot_runs],
                         [RunStatus.SUCCEEDED, RunStatus.QUEUED])
        return manifest

    def test_interrupted_partial_run_skips_completed_shot_and_replays_same_attempt(self):
        interrupted = self.interrupt_after_first_shot()
        artifact = interrupted.artifacts[0]
        before = (self.output / artifact.uri).read_bytes()
        provider = FakeExecutionProvider()
        with patch('cineforge.mock.FakeExecutionProvider', return_value=provider), \
                patch.object(provider, 'submit', wraps=provider.submit) as submit:
            result = execute_mock(self.plan, output_dir=self.output, resume=True)
            self.assertEqual(submit.call_count, 1)
            self.assertEqual(submit.call_args.args[0].shot.shot_id,
                             self.plan.plan.shots[1].shot_id)
        self.assertEqual(result.manifest.shot_runs[0], interrupted.shot_runs[0])
        self.assertEqual(result.manifest.shot_runs[1].job_id, interrupted.shot_runs[1].job_id)
        self.assertEqual(result.manifest.shot_runs[1].attempt, 1)
        self.assertEqual((self.output / artifact.uri).read_bytes(), before)

    def test_fresh_process_cli_resume_of_partial_run(self):
        partial = self.interrupt_after_first_shot()
        path = self.output / partial.artifacts[0].uri
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        command = [sys.executable, '-m', 'cineforge', 'mock-run', str(EXAMPLE),
                   '--output-dir', str(self.output), '--resume']
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        completed = RunManifest.from_json(result.stdout)
        self.assertTrue(all(run.status == RunStatus.SUCCEEDED for run in completed.shot_runs))
        self.assertEqual(completed.shot_runs[0], partial.shot_runs[0])
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        again = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertEqual(again.stdout, result.stdout)

    def test_atomic_write_and_replace_failures_preserve_old_checkpoint(self):
        for operation in ('fsync', 'replace'):
            with self.subTest(operation=operation):
                self.output = self.root / operation
                with LocalRunStore(self.output) as store:
                    manifest = store.start(self.plan, mock_run_id(self.plan), resume=False)
                    before = (self.output / 'checkpoint.json').read_bytes()
                    with patch(f'cineforge.storage.os.{operation}', side_effect=OSError('fault')):
                        with self.assertRaises(OSError):
                            store.checkpoint(self.plan, manifest)
                    self.assertEqual(before, (self.output / 'checkpoint.json').read_bytes())
                execute_mock(self.plan, output_dir=self.output, resume=True)

    def test_success_checkpoint_interruption_recovers_orphan_payload(self):
        original = LocalRunStore.checkpoint

        def checkpoint(store, plan, manifest):
            if manifest.artifacts:
                raise KeyboardInterrupt('before success checkpoint')
            return original(store, plan, manifest)

        with patch.object(LocalRunStore, 'checkpoint', new=checkpoint):
            with self.assertRaises(KeyboardInterrupt):
                execute_mock(self.plan, output_dir=self.output)
        orphans = list(self.output.glob('artifact-*.json'))
        self.assertEqual(len(orphans), 1)
        before = (orphans[0].read_bytes(), orphans[0].stat().st_mtime_ns)
        (self.output / ('.pending-' + '1' * 32)).write_bytes(b'interrupted write')
        result = execute_mock(self.plan, output_dir=self.output, resume=True)
        self.assertEqual(len(result.manifest.artifacts), 2)
        self.assertEqual(before, (orphans[0].read_bytes(), orphans[0].stat().st_mtime_ns))

    def test_lock_blocks_second_process_and_releases_after_exception(self):
        with LocalRunStore(self.output) as store:
            store.start(self.plan, mock_run_id(self.plan), resume=False)
            result = subprocess.run(
                [sys.executable, '-m', 'cineforge', 'mock-run', str(EXAMPLE),
                 '--output-dir', str(self.output), '--resume'],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, '')
            self.assertIn('locked', result.stderr)
        execute_mock(self.plan, output_dir=self.output, resume=True)

    def test_symlinks_hardlinks_and_overwrite_are_rejected(self):
        execute_mock(self.plan, output_dir=self.output)
        original = (self.output / 'checkpoint.json').read_bytes()
        with self.assertRaises(StorageError):
            execute_mock(self.plan, output_dir=self.output)
        self.assertEqual(original, (self.output / 'checkpoint.json').read_bytes())
        linked = self.root / 'linked'
        linked.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(OSError):
            execute_mock(self.plan, output_dir=linked, resume=True)
        artifact = self.output / self.checkpoint()['manifest']['artifacts'][0]['uri']
        outside = self.root / 'outside'
        artifact.rename(outside)
        artifact.symlink_to(outside)
        self.assert_rejected()
        self.assertEqual(outside.read_bytes(), artifact.read_bytes())
        artifact.unlink()
        os.link(outside, artifact)
        self.assert_rejected()

    def test_new_or_resume_preconditions(self):
        with self.assertRaises(StorageError):
            execute_mock(self.plan, output_dir=self.output, resume=True)
        with self.assertRaises(ValueError):
            execute_mock(self.plan, resume=True)
        (self.output / 'unrelated').write_bytes(b'keep')
        with self.assertRaises(StorageError):
            execute_mock(self.plan, output_dir=self.output)
        self.assertEqual((self.output / 'unrelated').read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
