import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts import code_evolution as evo
from scripts import evolve_per_user as per
from scripts.blackbox_harness import CompiledHarness, runtime_row, validate_harness_code, VLLMQA
from scripts.embedding_client import EmbeddingClient
from scripts.isolated_runtime import run_isolated


class FakeClient:
    def generate_batch(self, prompts, **kwargs):
        return ['generated answer' for _ in prompts]


class V2RuntimeTest(unittest.TestCase):
    def test_deep_copy(self):
        original = {'profile': [{'title': 'original', 'nested': {'a': 1}}], 'target': 'secret'}
        safe = runtime_row(original)
        safe['profile'][0]['nested']['a'] = 2
        self.assertEqual(original['profile'][0]['nested']['a'], 1)
        self.assertNotIn('target', safe)

    def test_normal_classes_and_public_qa(self):
        code = '''class Memory:
    def __init__(self, name):
        self.name = name
def run(row, qa):
    memory = Memory('test')
    assert not hasattr(qa, 'client')
    return memory.name + ':' + qa.generate(row['input'])
'''
        qa = VLLMQA(FakeClient())
        self.assertEqual(run_isolated(code, {'input': 'task'}, qa, timeout=10), 'test:generated answer')
        self.assertEqual(qa.calls, 1)

    def test_infinite_top_level_not_executed_by_parser_and_times_out(self):
        code = 'while True:\n    pass\ndef run(row, qa):\n    return "ok"'
        parsed, error = evo.parse_proposal(code, 'macro_strategy', 'def run(row, qa):\n    return "old"')
        self.assertIsNotNone(parsed, error)
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            run_isolated(code, {}, VLLMQA(FakeClient()), timeout=.4)
        self.assertLess(time.monotonic()-start, 3)

    def test_async_bad_signature_and_non_string_rejected(self):
        self.assertTrue(validate_harness_code('async def run(row, qa):\n    return "x"'))
        self.assertTrue(validate_harness_code('def run(row):\n    return "x"'))
        with self.assertRaises(TypeError):
            CompiledHarness('def run(row, qa):\n    return 42').run({}, None)

    def test_fresh_globals_each_sample(self):
        code = 'state = []\ndef run(row, qa):\n    state.append(1)\n    return str(len(state))'
        rows = [dict(user_id='u', sample_id=str(i), input='task', target='1', profile=[]) for i in range(3)]
        results = evo.evaluate_code(code, rows, FakeClient(), qa_max_calls=8, qa_concurrency=3, label='isolation')
        self.assertEqual([r['prediction'] for r in results], ['1', '1', '1'])

    def test_installed_science_and_io_guard(self):
        code = '''import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
def run(row, qa):
    g = nx.Graph()
    g.add_edge('a', 'b')
    score = cosine_similarity(csr_matrix(np.eye(2)))[0, 0]
    return str((nx.shortest_path(g, 'a', 'b'), float(score)))
'''
        self.assertEqual(run_isolated(code, {}, VLLMQA(None), 20), "(['a', 'b'], 1.0)")
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp)/'target.txt'
            secret.write_text('42')
            code = 'import numpy as np\ndef run(row, qa):\n    return str(np.loadtxt(row["input"]))'
            with self.assertRaisesRegex(RuntimeError, 'library resources'):
                run_isolated(code, {'input': str(secret)}, VLLMQA(None), 20)

    def test_generation_and_embedding_budgets(self):
        qa = VLLMQA(FakeClient(), max_calls=1)
        with self.assertRaisesRegex(RuntimeError, 'call budget'):
            run_isolated('def run(row, qa):\n    return str(qa.generate_many(["a", "b"]))', {}, qa, 10)
        qa = VLLMQA(None, embedding_client=object())
        with self.assertRaisesRegex(RuntimeError, 'embedding budget'):
            qa.embed(['x']*4097)

    def test_runtime_answer_generation_is_deterministic(self):
        calls = []

        class RecordingClient:
            def generate_batch(self, prompts, **kwargs):
                calls.append(kwargs)
                return ['answer'] * len(prompts)

        qa = VLLMQA(RecordingClient())
        qa.generate('task', temperature=.8, top_p=.2)
        qa.generate_many(['task-a', 'task-b'], temperature=.8, top_p=.2)
        self.assertEqual(calls[0]['temperature'], 0.0)
        self.assertEqual(calls[0]['top_p'], 1.0)
        self.assertEqual(calls[1]['temperature'], 0.0)
        self.assertEqual(calls[1]['top_p'], 1.0)


class V2FeedbackTest(unittest.TestCase):
    def test_gold_available_only_in_training_feedback_and_old_steps_represented(self):
        row = dict(user_id='u', sample_id='s', input='Title', target='training gold',
                   prediction='other words', profile=[], qa_trace=[])
        entries = evo.build_failure_entries([], [row], iteration=0, operation='seed', candidate_id='seed')
        self.assertNotIn('target', entries[0])
        self.assertNotIn('training gold', str(entries[0]))
        self.assertNotIn('task_entity_coverage_low', entries[0]['failure_types'])
        with tempfile.TemporaryDirectory() as tmp:
            bank = evo.FailureBank(Path(tmp)/'bank.jsonl')
            for i in range(30):
                bank.add([dict(entries[0], candidate_id=str(i), iteration=i)])
            view = bank.prompt_view(12)
            self.assertIn('0', [r['candidate_id'] for r in view])
            self.assertIn('29', [r['candidate_id'] for r in view])
            self.assertEqual(bank.recent(0), [])
            bank.add([dict(entries[0], candidate_id='29', iteration=29)])
            self.assertEqual(len(bank.entries), 30)

    def test_no_min_or_max_and_truncation_is_visible(self):
        def response(request, timeout):
            payload = json.loads(request.data)
            self.assertNotIn('min_tokens', payload)
            self.assertNotIn('max_tokens', payload)
            return io.BytesIO(json.dumps({'choices': [{'message': {'content':'partial'}, 'finish_reason':'length'}]}).encode())
        with patch('urllib.request.urlopen', side_effect=response):
            with self.assertRaisesRegex(RuntimeError, 'truncated'):
                evo.agent_chat('http://test/v1', 'Qwen/Qwen3.8-27B', 'prompt', timeout=1,
                               retries=1, retry_wait=0, max_tokens=None)

    def test_comment_only_change_is_noop(self):
        parent = 'def run(row, qa):\n    return "same"'
        proposal, error = evo.parse_proposal('# new idea\n'+parent, 'macro_strategy', parent)
        self.assertIsNone(proposal)
        self.assertIn('unchanged', error)

    def test_empty_response_retries_original_prompt(self):
        requests = []
        def response(request, timeout):
            requests.append(json.loads(request.data))
            content = '' if len(requests)==1 else 'def run(row, qa):\n    return "ok"'
            return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':content}}]}).encode())
        with patch('urllib.request.urlopen', side_effect=response):
            output = evo.agent_chat('http://test/v1', 'model', 'full training context', timeout=1,
                                    retries=2, retry_wait=0, max_tokens=None)
        self.assertIn('def run', output)
        self.assertEqual(requests[0], requests[1])

    def test_small_context_selects_whole_cases_keeps_scores_and_parent(self):
        cases = [dict(sample_id=str(i), input='task', target=('reference '+str(i)+' ')*400,
                      prediction='prediction '*400, scores_100={'rougeL':20+i}) for i in range(8)]
        parent = 'def run(row, qa):\n    return qa.generate(row["input"])'
        prompt = ('Step 1; operation=macro_strategy; branch=2.\nCurrent branch evaluation; comparisons:\n'
            +json.dumps({'worst_traces':cases})+'\n\nTraining feedback sampled ACROSS previous steps:\n[]\n\n'
            +'Evolution history (incumbent):\n[]\n\nHistorical alternatives with actual source/diffs:\n[]\n\n'
            +'Current branch source (replaceable):\n```python\n'+parent+'\n```')
        def response(request, timeout):
            if isinstance(request, str):
                return io.BytesIO(json.dumps({'data':[{'id':'Qwen/Qwen3.8-27B','max_model_len':16384}]}).encode())
            text=json.loads(request.data)['prompt']
            return io.BytesIO(json.dumps({'count':len(text)//3}).encode())
        records=[]
        with patch('urllib.request.urlopen', side_effect=response):
            fitted=evo.fit_agent_context(prompt,'http://test/v1',records.append)
        self.assertIn(parent,fitted)
        summary=json.loads(fitted.split('comparisons:\n')[1].split('Training feedback')[0])
        self.assertLess(len(summary['worst_traces']),8)
        self.assertEqual(len(summary['all_training_case_scores']),8)
        for case in summary['worst_traces']:
            self.assertEqual(case['target'],cases[int(case['sample_id'])]['target'])
        self.assertLessEqual(records[-1]['input_tokens']+records[-1]['output_space_reserved']+256,16384)

    def test_complete_reference_literal_rejected(self):
        target='A complete reference abstract with specific content. '*6
        code='def run(row, qa):\n    return '+repr(target)
        self.assertIsNotNone(evo.reference_literal_leak(code,[{'target':target}]))

    def test_short_label_tasks_reject_input_and_sample_lookup(self):
        row = {
            'input': 'A long current task description with enough content to identify one exact training sample. ' * 2,
            'sample_id': 'profile_adaptation:lamp2:120:1200',
            'target': 'healthy living',
        }
        input_code = 'def run(row, qa):\n    return '+repr(row['input'])
        sample_code = 'def run(row, qa):\n    return '+repr(row['sample_id'])
        self.assertIn('input lookup', evo.reference_literal_leak(input_code, [row]))
        self.assertIn('sample_id lookup', evo.reference_literal_leak(sample_code, [row]))

    def test_one_request_failure_does_not_drop_other_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = evo.FailureBank(Path(tmp)/'bank.jsonl')
            def request(url, model, prompt, **kwargs):
                if 'For this planning request ONLY' in prompt:
                    return "HYPOTHESES = ['different A', 'different B']"
                if 'Your hypothesis: different A' in prompt:
                    raise RuntimeError('temporary outage')
                return 'def run(row, qa):\n    return qa.generate(row["input"])'
            with patch.object(evo, 'agent_chat', side_effect=request):
                proposals = evo.propose_candidates(parent_code='def run(row, qa):\n    return "seed"',
                    operation='macro_strategy', iteration=1, count=2, current_summary={},
                    failure_bank=bank, history=[], archive=[], strategy_library='',
                    agent_api_url='unused', agent_api_model='model', agent_timeout=1,
                    agent_retries=1, agent_retry_wait=0, agent_concurrency=2, agent_max_tokens=None)
            self.assertFalse(proposals[0]['valid'])
            self.assertTrue(proposals[1]['valid'])

    def test_operation_planner_assigns_concrete_hypotheses(self):
        bank = evo.FailureBank(Path(tempfile.mkdtemp()) / 'bank.jsonl')
        prompts = []

        def request(url, model, prompt, **kwargs):
            prompts.append(prompt)
            return (
                '# HYPOTHESIS 1: test retrieval with title-aware diversity\n'
                '# HYPOTHESIS 2: test a compact style and content card\n'
                '# HYPOTHESIS 3: test a cautious draft revision\n'
                '# HYPOTHESIS 4: test a task-conditioned fallback\n'
            )

        with patch.object(evo, 'agent_chat', side_effect=request):
            hypotheses = evo.plan_hypotheses(
                parent_code='def run(row, qa):\n    return "seed"',
                operation='macro_strategy', iteration=2, count=4,
                current_summary={}, failure_bank=bank, history=[], archive=[],
                strategy_library='', agent_api_url='unused', agent_api_model='model',
                agent_timeout=1, agent_retries=1, agent_retry_wait=0,
                agent_max_tokens=None)
        self.assertEqual(len(hypotheses), 4)
        self.assertEqual(len(set(hypotheses)), 4)
        self.assertEqual(len(prompts), 1)

    def test_single_candidate_fallback_is_not_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = evo.FailureBank(Path(tmp) / 'bank.jsonl')

            def request(url, model, prompt, **kwargs):
                return 'def run(row, qa):\n    return qa.generate(row["input"])'

            with patch.object(evo, 'agent_chat', side_effect=request):
                proposals = evo.propose_candidates(
                    parent_code='def run(row, qa):\n    return "seed"',
                    operation='micro_repair', iteration=1, count=1,
                    current_summary={}, failure_bank=bank, history=[], archive=[],
                    strategy_library='', agent_api_url='unused', agent_api_model='model',
                    agent_timeout=1, agent_retries=1, agent_retry_wait=0,
                    agent_concurrency=1, agent_max_tokens=None)
        self.assertTrue(proposals[0]['valid'])
        self.assertNotIn('Independent branch', proposals[0]['hypothesis'])

    def test_rejected_macro_gets_micro_repair_and_can_win(self):
        macro = 'def run(row, qa):\n    return "macro"'
        repair = 'def run(row, qa):\n    return "repaired"'
        observed_parents = []
        def propose(**kwargs):
            observed_parents.append(kwargs['parent_code'])
            code = macro if kwargs['operation']=='macro_strategy' else repair if kwargs['parent_code']==macro else kwargs['parent_code']
            return [{'valid':True, 'code':code} for _ in range(kwargs['count'])]
        def evaluate(code, rows, qa, **kwargs):
            prediction = 'alpha' if code==macro else 'alpha beta gamma' if code==repair else 'alpha beta'
            return [dict(r, prediction=prediction, error='', qa_calls=1) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp), iterations=2, branches=3, agent_url='unused')
            rows=[dict(user_id='u', sample_id='s', input='task', target='alpha beta gamma', profile=[])]
            choices = [{'operation': op, 'reason': 'adaptation evidence'} for op in evo.OPERATION_NAMES]
            with patch.object(evo,'choose_operation',side_effect=choices), patch.object(evo,'propose_candidates',side_effect=propose), patch.object(evo,'evaluate_code',side_effect=evaluate):
                state=per.evolve_user('u',rows,args,None,threading.Semaphore(),'',lambda r:None)
            self.assertEqual(state['completed_operations'], 2)
            self.assertIn(macro, observed_parents)
            self.assertEqual(state['summary']['mean_rougeL'], 1)
            self.assertEqual(Path(state['code_path']).read_text(), repair)

    def test_operation_choice_is_data_only(self):
        for operation in evo.OPERATION_NAMES:
            self.assertEqual(evo.parse_operation_choice(
                f'operation = "{operation}"\nreason = "historical errors"')['operation'], operation)
        for text in ('operation = "both"\nreason = "x"',
                     'operation = "micro_repair"',
                     'import os\noperation = "macro_strategy"\nreason = "x"'):
            with self.assertRaises(ValueError):
                evo.parse_operation_choice(text)

    def test_rounds_can_repeat_operation_and_resume_without_more_work(self):
        calls = []
        def propose(**kwargs):
            calls.append((kwargs['iteration'], kwargs['operation'], kwargs['count']))
            return [{'valid': True, 'code': kwargs['parent_code']} for _ in range(kwargs['count'])]
        def evaluate(code, rows, qa, **kwargs):
            return [dict(r, prediction='alpha beta', error='', qa_calls=1) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output_dir=Path(tmp), iterations=2, branches=3, agent_url='unused')
            rows = [dict(user_id='u', sample_id='s', input='task', target='alpha beta', profile=[])]
            with patch.object(evo, 'choose_operation', return_value={'operation': 'micro_repair', 'reason': 'errors'}) as choose, patch.object(evo, 'propose_candidates', side_effect=propose), patch.object(evo, 'evaluate_code', side_effect=evaluate):
                state = per.evolve_user('u', rows, args, None, threading.Semaphore(), '', lambda r: None)
                self.assertEqual(state['completed_operations'], 2)
                self.assertEqual(calls, [(1, 'micro_repair', 3), (2, 'micro_repair', 3)])
                per.evolve_user('u', rows, args, None, threading.Semaphore(), '', lambda r: None)
                self.assertEqual(choose.call_count, 2)
                self.assertEqual(len(calls), 2)


class V2EmbeddingTest(unittest.TestCase):
    def test_warm_cache_not_blocked_by_cold_request(self):
        client = EmbeddingClient('http://unused')
        client.cache['warm'] = (1., 0.)
        entered, release = threading.Event(), threading.Event()
        def request(*args, **kwargs):
            entered.set()
            release.wait(3)
            return io.BytesIO(json.dumps({'data':[{'index':0,'embedding':[0.,1.]}]}).encode())
        with patch('urllib.request.urlopen', side_effect=request), ThreadPoolExecutor(2) as pool:
            cold = pool.submit(client.embed, ['cold'])
            self.assertTrue(entered.wait(1))
            try:
                warm = pool.submit(client.embed, ['warm'])
                self.assertEqual(warm.result(timeout=.5), [[1.,0.]])
            finally:
                release.set()
            self.assertEqual(cold.result(), [[0.,1.]])


if __name__ == '__main__':
    unittest.main()
