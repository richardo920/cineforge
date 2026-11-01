"""Synthetic CLI acceptance and complete-preflight probes; no external services."""

import contextlib
import copy
import io
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cineforge.cli import main
from cineforge.contracts import ContractError, ProjectPlan, RunManifest, RunStatus
from cineforge.mock import (
    MockPlan,
    execute_mock,
    load_mock_plan,
    mock_idempotency_key,
    mock_run_id,
)
from cineforge.providers import FakeExecutionProvider, ProviderError

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic-plan.json'


class MockCliTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(EXAMPLE.read_text(encoding='utf-8'))

    def invoke(self, command, raw=None, *options):
        with tempfile.TemporaryDirectory(prefix='cineforge-m14-test-') as directory:
            path = Path(directory) / 'plan.json'
            path.write_bytes((json.dumps(self.data) if raw is None else raw).encode('utf-8'))
            before = path.read_bytes()
            result = subprocess.run(
                [sys.executable, '-m', 'cineforge', command, str(path), *options],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            return result

    def assert_invalid(self, raw):
        for command in ('validate-plan', 'mock-run'):
            with self.subTest(command=command, raw=raw[:100]):
                result = self.invoke(command, raw)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertIn('error:', result.stderr)

    def test_validation_emits_canonical_plan(self):
        result = self.invoke('validate-plan')
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = ProjectPlan.from_json(result.stdout)
        self.assertEqual(result.stdout, plan.to_json() + '\n')
        self.assertEqual(len(plan.shots), 2)

    def test_mock_run_is_canonical_deterministic_metadata(self):
        first = self.invoke('mock-run')
        second = self.invoke('mock-run')
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        manifest = RunManifest.from_json(first.stdout)
        self.assertEqual(first.stdout, manifest.to_json() + '\n')
        self.assertEqual(len(manifest.shot_runs), 2)
        self.assertEqual(len(manifest.artifacts), 2)
        self.assertTrue(all(run.status == RunStatus.SUCCEEDED for run in manifest.shot_runs))
        self.assertTrue(all(run.cost_usd == 0 for run in manifest.shot_runs))
        self.assertTrue(all(artifact.uri.startswith('memory://cineforge.fake-v1/')
                            for artifact in manifest.artifacts))
        self.assertTrue(all(artifact.license == 'LicenseRef-SyntheticFixture'
                            for artifact in manifest.artifacts))

    def test_run_id_override(self):
        result = self.invoke('mock-run', None, '--run-id', 'demo-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(RunManifest.from_json(result.stdout).run_id, 'demo-run')
        self.assertNotEqual(result.stdout, self.invoke('mock-run').stdout)
        invalid = self.invoke('mock-run', None, '--run-id', '../invalid')
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, '')

    def test_strict_json(self):
        for raw in ('{', '[]', 'null', '{"schema_version":1,"schema_version":1}',
                    json.dumps(self.data).replace('"seed": 42', '"seed": NaN'),
                    json.dumps(self.data).replace('"seed": 42', '"seed": Infinity'),
                    '[' * 2000):
            self.assert_invalid(raw)

    def test_schema_and_metadata_rejections(self):
        cases = []
        for section, key, value in (
            (None, 'schema_version', True), (None, 'unknown', 1),
            ('plan', 'unknown', 1), ('models', 'revision', 'latest'),
            ('models', 'license', ''), ('models', 'commercial_use', 1),
            ('models', 'minimum_memory_mib', True),
            ('models', 'allowed_backends', ['cpu', 'cpu']),
            ('models', 'required_precision', 'fp64'),
            ('devices', 'backend', 'auto'), ('devices', 'memory_mib', -1),
            ('devices', 'supported_precisions', ['fp32', 'fp32']),
        ):
            data = copy.deepcopy(self.data)
            target = data if section is None else data[section]
            if isinstance(target, list):
                target = target[0]
            target[key] = value
            cases.append(data)
        for section in ('models', 'devices'):
            data = copy.deepcopy(self.data)
            data[section].append(copy.deepcopy(data[section][0]))
            cases.append(data)
            data = copy.deepcopy(self.data)
            data[section] = []
            cases.append(data)
        for data in cases:
            self.assert_invalid(json.dumps(data))

    def test_invalid_later_shot_and_unavailable_device(self):
        for key, value in (('scene_id', 'missing'), ('model_id', 'missing'),
                           ('model_revision', 'missing'), ('duration_seconds', 0),
                           ('prompt', '\ud800')):
            data = copy.deepcopy(self.data)
            data['plan']['shots'][1][key] = value
            self.assert_invalid(json.dumps(data))
        self.data['devices'][0]['memory_mib'] = 0
        self.assert_invalid(json.dumps(self.data))

    def test_file_and_help_errors(self):
        for args, code in (([], 0), (['--help'], 0), (['mock-run', '--help'], 0),
                           (['mock-run'], 2), (['validate-plan', '/missing/plan.json'], 2),
                           (['mock-run', str(EXAMPLE), '--unknown'], 2)):
            result = subprocess.run([sys.executable, '-m', 'cineforge', *args],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, code, result.stderr)
        with tempfile.TemporaryDirectory(prefix='cineforge-m14-utf8-') as directory:
            path = Path(directory) / 'invalid.json'
            path.write_bytes(b'\xff')
            result = subprocess.run([sys.executable, '-m', 'cineforge', 'mock-run', str(path)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, '')

    def test_complete_preflight_before_provider_creation(self):
        for field, value in (('scene_id', 'missing'), ('model_id', 'missing')):
            data = copy.deepcopy(self.data)
            data['plan']['shots'][1][field] = value
            with tempfile.TemporaryDirectory(prefix='cineforge-m14-preflight-') as directory:
                path = Path(directory) / 'plan.json'
                path.write_text(json.dumps(data), encoding='utf-8')
                with patch('cineforge.mock.FakeExecutionProvider') as provider, \
                        patch.object(sys, 'argv', ['cineforge', 'mock-run', str(path)]), \
                        contextlib.redirect_stdout(io.StringIO()) as stdout, \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as exit_result:
                        main()
                    self.assertEqual(exit_result.exception.code, 2)
                    self.assertEqual(stdout.getvalue(), '')
                    provider.assert_not_called()
        plan = load_mock_plan(EXAMPLE)
        with patch('cineforge.mock.FakeExecutionProvider') as provider:
            with self.assertRaises(ContractError):
                execute_mock(plan, run_id='bad/id')
            provider.assert_not_called()

    def test_no_network_subprocess_or_filesystem_generation(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network')), \
                patch.object(subprocess, 'Popen', side_effect=AssertionError('process')), \
                patch('builtins.open', side_effect=AssertionError('file generation')):
            plan = MockPlan.from_json(json.dumps(self.data))
            result = execute_mock(plan)
        self.assertEqual(len(result.results), 2)
        for output in result.results:
            payload = json.loads(output.payload)
            self.assertIs(payload['synthetic'], True)
            self.assertEqual(payload['request']['model']['license'], 'LicenseRef-SyntheticFixture')
            self.assertEqual(payload['request']['shot']['seed'],
                             plan.plan.get_shot(output.run.shot_id).seed)

    def test_full_declaration_identity_and_public_helpers(self):
        plan = load_mock_plan(EXAMPLE)
        self.assertEqual(plan.to_json(), MockPlan.from_json(plan.to_json()).to_json())
        requests = plan.preflight()
        identity = mock_run_id(plan)
        result = execute_mock(plan)
        self.assertEqual(result.manifest.run_id, identity)
        self.assertEqual(result.manifest.shot_runs[0].idempotency_key,
                         mock_idempotency_key(identity, requests[0]))
        for section, field, value in (('models', 'license', 'LicenseRef-Changed'),
                                      ('devices', 'memory_mib', 2)):
            data = copy.deepcopy(self.data)
            data[section][0][field] = value
            self.assertNotEqual(mock_run_id(MockPlan.from_json(json.dumps(data))), identity)
        data = copy.deepcopy(self.data)
        unused = dict(data['models'][0], model_id='unused-model')
        data['models'].append(unused)
        additional = MockPlan.from_json(json.dumps(data))
        self.assertEqual(additional.to_dict()['models'][-1], unused)
        self.assertNotEqual(mock_run_id(additional), identity)

    def test_later_incompatible_request_prevents_all_submissions(self):
        data = copy.deepcopy(self.data)
        data['models'].append(dict(data['models'][0], model_id='too-large', minimum_memory_mib=2))
        data['plan']['shots'][1]['model_id'] = 'too-large'
        with patch('cineforge.mock.FakeExecutionProvider') as provider:
            with self.assertRaisesRegex(ContractError, 'no compatible device'):
                MockPlan.from_json(json.dumps(data))
            provider.assert_not_called()
        self.assert_invalid(json.dumps(data))

    def test_failure_and_polling_are_bounded(self):
        plan = load_mock_plan(EXAMPLE)
        with patch('cineforge.mock.FakeExecutionProvider',
                   return_value=FakeExecutionProvider(fail=True)):
            with self.assertRaises(ProviderError):
                execute_mock(plan)
        provider = FakeExecutionProvider()
        with patch('cineforge.mock.FakeExecutionProvider', return_value=provider), \
                patch.object(provider, 'status') as status, patch.object(provider, 'cancel') as cancel:
            status.return_value.status = RunStatus.RUNNING
            with self.assertRaises(ProviderError):
                execute_mock(plan)
            self.assertEqual(status.call_count, 2)
            cancel.assert_called_once()


if __name__ == '__main__':
    unittest.main()
