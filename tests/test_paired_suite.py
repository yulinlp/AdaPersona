import argparse
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts import evolve_per_user as per
from scripts import code_evolution as evo
from scripts.monitor_lamp_test import discover_events
from scripts.lamp_tasks import LampTaskAdapter
from scripts.paired_suite import BASELINE, METHODS
from scripts.blackbox_harness import CompiledHarness, runtime_row


class PairedSuiteTest(unittest.TestCase):
    def test_concurrent_metric_preflight_in_fresh_process(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, '-c',
            'from concurrent.futures import ThreadPoolExecutor; '
            'from scripts.evolution_metrics import preflight; '
            'pool=ThreadPoolExecutor(10); '
            'results=list(pool.map(lambda _: preflight(), range(20))); '
            'assert len(results)==20; pool.shutdown()'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_baselines_never_receive_gold_and_use_public_qa(self):
        class QA:
            def generate(self, prompt, **kwargs):
                assert 'SECRET_GOLD' not in prompt
                return 'answer'
        row = runtime_row(dict(user_id='u', sample_id='s', input='task', target='SECRET_GOLD',
                               profile=[dict(text='history', title='old title')]))
        for method in METHODS:
            code = f'METHOD={method!r}\nOUTPUT_TOKENS=128\n' + BASELINE
            self.assertEqual(CompiledHarness(code).run(row, QA()), 'answer')

    def test_native_objectives_and_invalid_ratings(self):
        task2 = LampTaskAdapter(2)
        self.assertEqual(task2.weighted_score(task2.row_metrics(
            dict(prediction='healthy', target='healthy living'))), 0)
        task3 = LampTaskAdapter(3)
        rows = [dict(user_id='u', input='', prediction='4', target='5'),
                dict(user_id='u', input='', prediction='1 or 5', target='5')]
        result = task3.summarize_rows(rows)
        self.assertEqual(result['invalid_predictions'], 1)
        self.assertEqual(result['mae'], 2.5)
        self.assertEqual(result['weighted_score'], 1 - result['mae']/4)

    def test_commit_recovers_missing_monitor_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output_dir=Path(tmp), iterations=1, branches=1, agent_url='unused')
            rows = [dict(user_id='u', sample_id='s', input='task', target='alpha', profile=[])]
            def evaluate(code, items, qa, **kwargs):
                return [dict(r, prediction='alpha', error='', qa_calls=1) for r in items]
            with patch.object(evo, 'evaluate_code', side_effect=evaluate), patch.object(
                    evo, 'choose_operation', return_value=dict(operation='micro_repair', reason='history')), patch.object(
                    evo, 'propose_candidates', return_value=[]):
                per.evolve_user('u', rows, args, None, threading.Semaphore(), '', lambda _: None)
                folder = Path(tmp)/'users'/per.user_key('u')
                # Simulate failure after state commit but before journal publication.
                (folder/'history.jsonl').write_text('')
                events = discover_events(Path(tmp), {'u': rows})
                self.assertEqual([e['iteration'] for e in events], [0, 1])
                per.evolve_user('u', rows, args, None, threading.Semaphore(), '', lambda _: None)
                history = evo.load_jsonl(folder/'history.jsonl')
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]['iteration'], 1)

    def test_selector_context_contract_and_identity_scrubbing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = evo.FailureBank(Path(tmp)/'bank.jsonl')
            def fit(prompt, *args):
                for marker in ('Current branch evaluation;', 'Training feedback sampled ACROSS',
                               'Evolution history (', 'Current branch source (replaceable):'):
                    self.assertIn(marker, prompt)
                self.assertNotIn('PRIVATE_USER', prompt)
                self.assertNotIn('operation = "macro_strategy"', prompt)
                return prompt
            with patch.object(evo, 'fit_agent_context', side_effect=fit), patch.object(
                    evo, 'agent_chat', return_value='operation="micro_repair"\nreason="historical failures"'):
                decision = evo.choose_operation(parent_code='def run(row, qa): return "x"', iteration=1,
                    current_summary={'worst_traces': [{'user_id': 'PRIVATE_USER', 'target': 'old'}]},
                    failure_bank=bank, history=[], archive=[], task_context='task', agent_api_url='unused',
                    agent_api_model='Qwen/Qwen3.8-27B', agent_timeout=1, request_gate=None, artifact_dir=Path(tmp))
                self.assertEqual(decision['operation'], 'micro_repair')

if __name__ == '__main__':
    unittest.main()
