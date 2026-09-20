import argparse
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from scripts import code_evolution as evo
from scripts import evolve_per_user as per
from scripts.profile_protocol import abstract_input, build_history_partitions, history_key, task_hint_loss


class ProfileProtocolTest(unittest.TestCase):
    def test_phrase_hints_and_conservation(self):
        text = abstract_input('Graph algorithms', 'We study bipartite permutation graphs and parallel scheduling algorithms.')
        self.assertIn('bipartite permutation graphs', text)
        row = dict(input=text, input_contract='preserve_task_hints')
        self.assertEqual(task_hint_loss(row, [{'prompt': text}]), [])
        self.assertTrue(task_hint_loss(row, [{'prompt': 'Graph algorithms: bipartite; parallel'}]))

    def test_validation_examples_never_enter_fit_profiles(self):
        profile = [dict(id=str(i), abstract=f'Parallel scheduling algorithms number {i}', title=f'Unique title number {i}') for i in range(16)]
        profile.append(dict(profile[0], id='duplicate'))
        profile.append(dict(id='missing', abstract='No abstract available.', title='Missing source'))
        fit, held, audit = build_history_partitions('u', profile, 'lamp', 5)
        self.assertEqual((len(fit), len(held), audit['usable_history']), (8,4,16))
        forbidden = {r['historical_key'] for r in held}
        for row in fit + held:
            self.assertFalse(forbidden & {history_key(item) for item in row['profile']})
            self.assertNotIn(row['historical_key'], {history_key(item) for item in row['profile']})
        self.assertEqual(build_history_partitions('u', profile, 'lamp', 5), (fit, held, audit))

    def test_short_profiles_still_have_disjoint_selection(self):
        profile = [dict(id=str(i), text=f'news {i}', title=f'headline {i}') for i in range(3)]
        fit, held, _ = build_history_partitions('u', profile, 'lamp', 4)
        self.assertEqual((len(fit), len(held)), (2,1))

    def test_short_generative_reference_literals_rejected(self):
        target = 'A new approach to network scheduling'
        rows = [dict(target=target, input='task', sample_id='s', benchmark='lamp', task=5)]
        self.assertIsNotNone(evo.reference_literal_leak(f'def run(row, qa): return {target!r}', rows))
        rows[0].update(target='food & drink', task=2)
        self.assertIsNone(evo.reference_literal_leak("def run(row, qa): return 'food & drink'", rows))

    def test_operation_parser_tolerates_plain_preamble(self):
        result = evo.parse_operation_choice('Two Python literal assignments:\n\noperation="micro_repair"\nreason="historical evidence"')
        self.assertEqual(result['operation'], 'micro_repair')

    def test_validation_regression_rejects_training_gain(self):
        candidate = 'def run(row, qa): return "candidate"'
        fit = [dict(user_id='u', sample_id='fit', input='task', target='alpha beta gamma', profile=[])]
        held = [dict(user_id='u', sample_id='held', input='hidden task', target='secret held reference', profile=[])]
        def evaluate(code, rows, qa, **kwargs):
            return [dict(r, prediction=('alpha beta gamma' if code==candidate else 'alpha beta')
                         if r['sample_id']=='fit' else ('wrong' if code==candidate else r['target']),
                         error='', qa_calls=1) for r in rows]
        def propose(**kwargs):
            self.assertNotIn('secret held reference', json.dumps(kwargs['current_summary']))
            self.assertNotIn('secret held reference', json.dumps(kwargs['failure_bank'].prompt_view()))
            return [dict(valid=True, code=candidate)]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp), iterations=1, branches=1, agent_url='unused')
            with patch.object(evo, 'evaluate_code', side_effect=evaluate), patch.object(evo, 'propose_candidates', side_effect=propose), patch.object(evo, 'choose_operation', return_value=dict(operation='micro_repair',reason='fit evidence')):
                state=per.evolve_user('u',fit,args,None,threading.Semaphore(),'',lambda r:None,selection_rows=held)
            self.assertEqual(Path(state['code_path']).name, 'seed.py')
            self.assertEqual(state['completed_operations'], 1)

    def test_seed_pool_selected_from_history_not_test(self):
        fit = [dict(user_id='u',sample_id='fit',input='task',target='alpha beta gamma',profile=[])]
        held = [dict(user_id='u',sample_id='held',input='private',target='secret historical label',profile=[])]
        def evaluate(code, rows, qa, **kwargs):
            return [dict(r, prediction=r['target'] if "METHOD='rag'" in code else 'wrong', error='',qa_calls=1) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp),iterations=0,branches=1,agent_url='unused',seed_pool=True)
            with patch.object(evo,'evaluate_code',side_effect=evaluate):
                state=per.evolve_user('u',fit,args,None,threading.Semaphore(),'',lambda r:None,selection_rows=held)
            self.assertEqual(state['seed_name'], 'rag')
            bank=(Path(tmp)/'users'/per.user_key('u')/'failure_bank.jsonl').read_text()
            self.assertNotIn('secret historical label', bank)
            self.assertEqual(len(evo.load_jsonl(Path(tmp)/'users'/per.user_key('u')/'archive.jsonl')), 5)

    def test_fit_plateau_and_selection_improvement_can_be_accepted(self):
        child='def run(row, qa): return "improved"'
        fit=[dict(user_id='u',sample_id='fit',input='fit task',target='alpha beta gamma',profile=[])]
        held=[dict(user_id='u',sample_id='held',input='held task',target='private validation reference',profile=[])]
        def evaluate(code, rows, qa, **kwargs):
            return [dict(r,prediction=r['target'] if code==child or r['sample_id']=='fit' else 'wrong',error='',qa_calls=1) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp),iterations=1,branches=1,agent_url='unused')
            with patch.object(evo,'evaluate_code',side_effect=evaluate), patch.object(evo,'propose_candidates',return_value=[dict(valid=True,code=child)]), patch.object(evo,'choose_operation',return_value=dict(operation='micro_repair',reason='fit evidence')):
                state=per.evolve_user('u',fit,args,None,threading.Semaphore(),'',lambda r:None,selection_rows=held)
            self.assertEqual(Path(state['code_path']).read_text(),child)

    def test_unstable_seed_blocks_formal_evolution(self):
        fit=[dict(user_id='u',sample_id='fit',input='task',target='alpha beta gamma',profile=[])]
        held=[dict(user_id='u',sample_id='held',input='private',target='hidden reference',profile=[])]
        def evaluate(code, rows, qa, **kwargs):
            return [dict(r,prediction='changed' if kwargs['label'].endswith(':seed') else r['target'],error='',qa_calls=1) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp),iterations=1,branches=1,agent_url='unused',seed_pool=True)
            with patch.object(evo,'evaluate_code',side_effect=evaluate), patch.object(evo,'propose_candidates') as proposer:
                with self.assertRaisesRegex(RuntimeError,'not reproducible'):
                    per.evolve_user('u',fit,args,None,threading.Semaphore(),'',lambda r:None,selection_rows=held)
                proposer.assert_not_called()

    def test_fit_and_selection_improvement_can_be_accepted(self):
        child='def run(row, qa): return "improved"'
        fit=[dict(user_id='u',sample_id='fit',input='fit task',target='alpha beta gamma',profile=[])]
        held=[dict(user_id='u',sample_id='held',input='held task',target='private validation reference',profile=[])]
        def evaluate(code, rows, qa, **kwargs):
            return [dict(r,prediction=r['target'] if code==child else r['target'].split()[0],error='',qa_calls=1) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp),iterations=1,branches=1,agent_url='unused')
            with patch.object(evo,'evaluate_code',side_effect=evaluate), patch.object(evo,'propose_candidates',return_value=[dict(valid=True,code=child)]), patch.object(evo,'choose_operation',return_value=dict(operation='micro_repair',reason='fit evidence')):
                state=per.evolve_user('u',fit,args,None,threading.Semaphore(),'',lambda r:None,selection_rows=held)
            self.assertEqual(Path(state['code_path']).read_text(),child)
            exact = per.LongLaMPAdapter().summarize_rows([dict(held[0], prediction=held[0]['target'], error='')])
            self.assertAlmostEqual(state['selection_summary']['weighted_score'], exact['weighted_score'])
            history=(Path(tmp)/'users'/per.user_key('u')/'history.jsonl').read_text()
            self.assertNotIn('private validation reference',history)

    def test_larger_history_budget_is_unique_and_disjoint(self):
        profile = [dict(id=str(i), text=f'news {i}', title=f'headline {i}') for i in range(48)]
        fit, held, _ = build_history_partitions('u', profile, 'lamp', 4, fit_limit=16, selection_limit=8)
        self.assertEqual((len(fit), len(held)), (16,8))
        self.assertEqual(len({r['historical_key'] for r in fit+held}), 24)
        forbidden = {r['historical_key'] for r in held}
        self.assertTrue(all(not forbidden.intersection(history_key(p) for p in r['profile']) for r in fit+held))
        with self.assertRaises(ValueError):
            build_history_partitions('u', profile, 'lamp', 4, fit_limit=0)

    def test_changed_intermediate_response_vetoes_even_same_score(self):
        child='def run(row, qa): return "improved"'
        fit=[dict(user_id='u',sample_id='fit',input='fit task',target='alpha beta gamma',profile=[])]
        held=[dict(user_id='u',sample_id='held',input='held task',target='private validation reference',profile=[])]
        def evaluate(code, rows, qa, **kwargs):
            response = 'different' if code==child and kwargs['label'].endswith(':confirm') else 'same'
            return [dict(r,prediction=r['target'] if code==child else 'wrong',error='',qa_calls=1,
                         qa_trace=[dict(prompt='fixed',response=response)]) for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(output_dir=Path(tmp),iterations=1,branches=1,agent_url='unused')
            with patch.object(evo,'evaluate_code',side_effect=evaluate), patch.object(evo,'propose_candidates',return_value=[dict(valid=True,code=child)]), patch.object(evo,'choose_operation',return_value=dict(operation='micro_repair',reason='fit evidence')):
                state=per.evolve_user('u',fit,args,None,threading.Semaphore(),'',lambda r:None,selection_rows=held)
            self.assertEqual(Path(state['code_path']).name,'seed.py')
            admission=json.loads((Path(tmp)/'users'/per.user_key('u')/'i01_micro_repair_b0_admission.json').read_text())
            self.assertEqual(admission['reason'],'execution_instability')

    def test_decision_format_repair_does_not_force_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank=evo.FailureBank(Path(tmp)/'bank.jsonl')
            responses=['I prefer micro repair based on the failure bank.',
                       'operation="micro_repair"\nreason="historical failures"']
            with patch.object(evo,'fit_agent_context',side_effect=lambda prompt,*a:prompt), patch.object(evo,'agent_chat',side_effect=responses) as request:
                choice=evo.choose_operation(parent_code='def run(row,qa): return "x"',iteration=1,
                    current_summary={},failure_bank=bank,history=[],archive=[],task_context='task',
                    agent_api_url='unused',agent_api_model='model',agent_timeout=1,request_gate=None,artifact_dir=Path(tmp))
            self.assertEqual(choice['operation'],'micro_repair')
            self.assertEqual(request.call_count,2)
            self.assertIn('Preserve its chosen operation',request.call_args.args[2])

if __name__ == '__main__':
    unittest.main()
