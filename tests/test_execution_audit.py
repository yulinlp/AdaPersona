import unittest
from scripts.execution_audit import compare_executions, historical_improvement
from scripts.lamp_tasks import LampTaskAdapter
from scripts.isolated_runtime import run_isolated
from scripts.blackbox_harness import VLLMQA


class ExecutionAuditTest(unittest.TestCase):
    def test_plateau_can_improve_selection_but_no_tradeoff_or_tie(self):
        adapter = LampTaskAdapter(2)
        def score(x): return {'mean_accuracy': x, 'errors': 0}
        self.assertTrue(historical_improvement(adapter, score(1), score(1), score(.75), score(.5)))
        self.assertFalse(historical_improvement(adapter, score(.9), score(1), score(.75), score(.5)))
        self.assertFalse(historical_improvement(adapter, score(1), score(1), score(.5), score(.5)))
        self.assertFalse(historical_improvement(adapter, score(1), score(.5), score(.5), score(.5)))
        self.assertFalse(historical_improvement(adapter, score(1), score(.5), score(.5), None))
        self.assertFalse(historical_improvement(adapter, score(1), score(.9), score(.4), score(.5)))
        self.assertFalse(historical_improvement(adapter, dict(score(1), errors=1), score(.9)))

    def test_first_divergence_not_downstream_prompt(self):
        def row(trace, prediction='ok'):
            return dict(user_id='u', sample_id='1', prediction=prediction, qa_trace=trace)
        a = row([dict(prompt='profile', response='A'), dict(prompt='A', response='ok')])
        b = row([dict(prompt='profile', response='B'), dict(prompt='B', response='ok')])
        audit = compare_executions([a], [b])
        self.assertFalse(audit['stable'])
        self.assertEqual(audit['changes'][0]['cause'], 'same_request_different_response')
        self.assertTrue(compare_executions([a], [a])['stable'])
        self.assertFalse(compare_executions([a], [])['stable'])
        with self.assertRaises(ValueError): compare_executions([a, a], [a])

    def test_isolated_hash_iteration_repeats(self):
        code = "def run(row, qa): return str(list(set(['alpha','beta','gamma','delta','epsilon'])))"
        values = [run_isolated(code, {}, VLLMQA(None), 10) for _ in range(4)]
        self.assertEqual(len(set(values)), 1)
