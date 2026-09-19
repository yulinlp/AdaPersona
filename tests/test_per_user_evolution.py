import argparse
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from scripts import evolve_per_user as per


class PerUserTest(unittest.TestCase):
    def setUp(self):
        chooser = patch.object(per.evo, 'choose_operation', return_value={
            'operation': 'macro_strategy', 'reason': 'Historical adaptation evidence'})
        chooser.start()
        self.addCleanup(chooser.stop)

    def test_independent_acceptance_failure_banks_and_resume(self):
        seen = []
        proposal = {'valid': True, 'code': "def run(row, qa):\n    return 'alpha'"}
        def evaluate(code, rows, qa, **kwargs):
            seen.append({r['user_id'] for r in rows})
            self.assertEqual(len(seen[-1]), 1)
            return [dict(r, prediction=('alpha' if code == proposal['code'] else 'beta'),
                         qa_calls=1, error='') for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output_dir=Path(tmp), iterations=1, branches=1, agent_url='unused')
            reports = []
            users = {'A': [{'user_id':'A', 'sample_id':'a', 'input':'alpha task',
                            'target':'alpha', 'profile':[]}],
                     'B': [{'user_id':'B', 'sample_id':'b', 'input':'beta task',
                            'target':'beta', 'profile':[]}]}
            with patch.object(per.evo, 'evaluate_code', side_effect=evaluate), patch.object(
                    per.evo, 'propose_candidates', return_value=[proposal]):
                states = {u:per.evolve_user(u, rows, args, None, threading.Semaphore(), '', reports.append)
                          for u,rows in users.items()}
            self.assertEqual(states['A']['summary']['mean_rougeL'], 1)
            self.assertEqual(states['B']['summary']['mean_rougeL'], 1)
            self.assertNotEqual(Path(states['A']['code_path']).name, 'seed.py')
            self.assertEqual(Path(states['B']['code_path']).name, 'seed.py')
            for u in users:
                directory = Path(tmp) / 'users' / per.user_key(u)
                entries = per.evo.load_jsonl(directory / 'failure_bank.jsonl') if (directory/'failure_bank.jsonl').exists() else []
                self.assertTrue(all(e.get('user_id') == u for e in entries))
                with patch.object(per.evo, 'evaluate_code') as evaluate_mock, patch.object(per.evo, 'propose_candidates') as propose_mock:
                    per.evolve_user(u, users[u], args, None, threading.Semaphore(), '', reports.append)
                    evaluate_mock.assert_not_called()
                    propose_mock.assert_not_called()

    def test_mixed_users_rejected_before_evaluation(self):
        with self.assertRaisesRegex(ValueError, 'isolation'):
            per.evolve_user('A', [{'user_id':'B'}], None, None, None, '', None)

    def test_runtime_errors_cannot_win(self):
        self.assertFalse(per.wins({'errors':1, 'mean_rougeL':1}, {'mean_rougeL':0}))

    def test_runtime_smoke_failure_skips_full_candidate_evaluation(self):
        proposal = {'valid': True, 'code': "def run(row, qa):\n    return 'bad'"}
        calls = []

        def evaluate(code, rows, qa, **kwargs):
            calls.append((code, len(rows)))
            if code == proposal['code']:
                return [dict(rows[0], prediction='', error='RuntimeError: AttributeError: list has no attribute items', qa_calls=0)]
            return [dict(row, prediction='alpha', error='', qa_calls=0) for row in rows]

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output_dir=Path(tmp), iterations=1, branches=1,
                                      agent_url='unused')
            rows = [{'user_id': 'A', 'sample_id': 'a', 'input': 'task',
                     'target': 'alpha', 'profile': []}]
            with patch.object(per.evo, 'evaluate_code', side_effect=evaluate), patch.object(
                    per.evo, 'propose_candidates', return_value=[proposal]):
                state = per.evolve_user('A', rows, args, None, threading.Semaphore(), '', lambda _: None)
            # One seed plus one chosen operation's smoke execution; no full pass.
            self.assertEqual([size for _, size in calls], [1, 1])
            directory = Path(tmp) / 'users' / per.user_key('A')
            self.assertTrue((directory / 'i01_macro_strategy_b0_smoke_failed.json').exists())
            self.assertFalse((directory / 'i01_micro_repair_b0_smoke_failed.json').exists())
            self.assertEqual(Path(state['code_path']).name, 'seed.py')

    def test_acceptance_uses_weighted_four_metric_objective(self):
        parent = {
            'errors': 0, 'mean_rouge1': .50, 'mean_rougeL': .50,
            'mean_bleu': .50, 'mean_meteor': .50,
        }
        # ROUGE-L alone improves, but the aggregate gets worse.
        child = {
            'errors': 0, 'mean_rouge1': .20, 'mean_rougeL': .60,
            'mean_bleu': .20, 'mean_meteor': .20,
        }
        self.assertFalse(per.wins(child, parent))
        child.update(mean_rouge1=.60, mean_bleu=.60, mean_meteor=.60)
        self.assertTrue(per.wins(child, parent))

if __name__ == '__main__':
    unittest.main()
