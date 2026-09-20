"""Independent code RSI per training user. No meta-policy or held-out evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import os
import shutil
import fcntl
import statistics
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from . import code_evolution as evo
    from .embedding_client import EmbeddingClient
    from .search_policy import select_parent_branches
    from .lamp_tasks import LampTaskAdapter
    from .execution_audit import compare_executions, historical_improvement
    from .profile_protocol import validate_history_partitions
except ImportError:
    import code_evolution as evo
    from embedding_client import EmbeddingClient
    from search_policy import select_parent_branches
    from lamp_tasks import LampTaskAdapter
    from execution_audit import compare_executions, historical_improvement
    from profile_protocol import validate_history_partitions


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def user_key(user_id):
    return hashlib.sha256(user_id.encode()).hexdigest()[:20]


def user_groups(rows):
    groups = evo.grouped(rows)
    if '' in groups:
        raise ValueError('Every training row must have a user_id')
    return groups


def wins(child, parent):
    # A different user's score can never enter this decision. Errors cannot
    # be traded against gains on the remaining examples.
    return child['errors'] == 0 and evo.weighted_score(child) > evo.weighted_score(parent) + 1e-12


PROTOCOL = 'per-user-code-v3-weighted'


class LongLaMPAdapter:
    """Compatibility adapter preserving the existing LongLaMP protocol."""

    benchmark = 'LongLaMP'
    task = 'abstract_generation'
    name = 'longlamp_abstract'
    seed_code = evo.SEED_HARNESS_CODE
    metric_names = ('rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor')
    metric_protocol = evo.METRIC_PROTOCOL
    objective_protocol = evo.OBJECTIVE_PROTOCOL
    objective_weights = evo.OBJECTIVE_WEIGHTS
    task_context = (
        'Benchmark: LongLaMP abstract generation. The harness receives a current abstract task '
        'and historical title/abstract dictionaries and must return only the current abstract.'
    )
    objective_description = (
        "maximize this user's weighted training score (0-100): "
        '0.25*ROUGE-1 + 0.35*ROUGE-L + 0.20*BLEU + 0.20*METEOR'
    )
    evaluation_boundary = ('Only historical-profile leave-one-out tasks are used for adaptation. '
                          'The current official test input, target and monitoring scores are unavailable to the evolver.')

    def preflight(self):
        module = importlib.import_module(evo.row_metrics.__module__)
        return module.preflight()

    def summarize_rows(self, rows, trace_limit=20):
        return evo.summarize_rows(rows, trace_limit=trace_limit)

    def build_failure_entries(self, parent_rows, child_rows, **kwargs):
        return evo.build_failure_entries(parent_rows, child_rows, **kwargs)

    def weighted_score(self, summary):
        return evo.weighted_score(summary)

    def score_key(self, summary):
        return evo.score_key(summary)

    def wins(self, child, parent):
        return wins(child, parent)


def adapter_for(args):
    if getattr(args, 'benchmark', 'longlamp') == 'lamp':
        return LampTaskAdapter(int(getattr(args, 'task', 4)))
    return LongLaMPAdapter()


def protocol_for(args):
    """Give the tree-search run a distinct, auditable protocol label."""
    agent_model = getattr(args, 'agent_model', 'Qwen/Qwen3.8-27B')
    qa_model = getattr(args, 'qa_model', 'Qwen2.5-7B-Instruct')
    agent_tag = agent_model.replace('/', '_').replace('-', '_')
    qa_tag = qa_model.replace('/', '_').replace('-', '_')
    benchmark = getattr(args, 'benchmark', 'longlamp')
    task = getattr(args, 'task', 'abstract_generation')
    prefix = f'{benchmark}-{task}-history-hidden-gain-v3'
    if getattr(args, 'search_strategy', 'greedy') == 'archive_beam':
        return f'{prefix}-per-user-code-v5-archive-beam-planned-smoke-weighted-agent-{agent_tag}-qa-{qa_tag}'
    return f'{prefix}-{PROTOCOL}-agent-{agent_tag}-qa-{qa_tag}'


def tools_reference(packages):
    return evo.load_strategy_library() + '\nInstalled optional package versions: ' + json.dumps(packages)


def evolve_user(user_id, rows, args, qa, gate, library, report, adapter=None, selection_rows=None):
    adapter = adapter or adapter_for(args)
    if not rows or {str(r['user_id']) for r in rows} != {user_id}:
        raise ValueError('User isolation violation')
    if not 1 <= args.branches <= 3:
        raise ValueError('Each round supports at most three candidates')
    selection_rows = list(selection_rows or [])
    validate_history_partitions(rows, selection_rows)
    if selection_rows and ({str(r['user_id']) for r in selection_rows} != {user_id} or
            {r['sample_id'] for r in rows} & {r['sample_id'] for r in selection_rows}):
        raise ValueError('History selection isolation violation')
    directory = args.output_dir / 'users' / user_key(user_id)
    protocol = protocol_for(args)
    agent_model = getattr(args, 'agent_model', 'Qwen/Qwen3.8-27B')
    agent_timeout = getattr(args, 'agent_timeout', 1200)
    qa_model = getattr(args, 'qa_model', 'Qwen2.5-7B-Instruct')
    qa_context = int(getattr(args, 'qa_context', 8192))
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / 'state.json'
    bank = evo.FailureBank.load(directory / 'failure_bank.jsonl')
    history_path, archive_path = directory / 'history.jsonl', directory / 'archive.jsonl'
    history = evo.load_jsonl(history_path) if history_path.exists() else []
    archive = evo.load_jsonl(archive_path) if archive_path.exists() else []
    dataset_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps({
        'protocol': protocol, 'benchmark': getattr(args, 'benchmark', 'longlamp'),
        'task': getattr(args, 'task', 'abstract_generation'),
        'metrics': adapter.metric_protocol,
        'objective_protocol': adapter.objective_protocol,
        'objective_weights': adapter.objective_weights, 'dataset': dataset_hash,
        'history_selection': selection_rows, 'seed_pool': getattr(args, 'seed_pool', False),
        'iterations': args.iterations, 'branches': args.branches,
        'search_strategy': getattr(args, 'search_strategy', 'greedy'),
        'beam_width': getattr(args, 'beam_width', 1),
        'archive_size': getattr(args, 'archive_size', 0),
        'island_count': getattr(args, 'island_count', 1),
        'incumbent_slots': getattr(args, 'incumbent_slots', 1),
        'qa_url': getattr(args, 'qa_url', ''), 'agent_url': args.agent_url,
        'agent_model': getattr(args, 'agent_model', 'Qwen/Qwen3.8-27B'),
        'qa_model': qa_model,
        'embedding_url': getattr(args, 'embedding_url', ''),
        'embedding_backend': getattr(args, 'embedding_backend', ''),
        'library': library, 'source_hash': getattr(args, 'source_hash', ''),
    }, sort_keys=True).encode()).hexdigest()

    def evaluate(code, label, eval_rows=None):
        target_rows = rows if eval_rows is None else eval_rows
        return evo.evaluate_code(code, target_rows, qa, qa_max_calls=8,
                                 qa_concurrency=min(8, len(target_rows)),
                                 sample_timeout=getattr(args, 'sample_timeout', 300),
                                 label=f'{user_key(user_id)}:{label}')

    def save_event(event):
        # Per-candidate checkpoints allow an interrupted operation to resume.
        history[:] = [r for r in history if not (
            r.get('event') == event.get('event') and
            r.get('candidate_id') == event.get('candidate_id') and
            r.get('iteration') == event.get('iteration') and
            r.get('operation') == event.get('operation'))]
        history.append(event)
        evo.write_jsonl(history_path, history)

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('config_hash') != config_hash or state['user_id'] != user_id:
            raise ValueError('Resume protocol/data/config/source differs; use a fresh output directory')
    else:
        code = adapter.seed_code
        seed_name = 'retrieval_seed'
        if getattr(args, 'seed_pool', False):
            if not selection_rows:
                raise ValueError('Multiple seed selection requires independent history-selection rows')
            from scripts.paired_suite import BASELINE, METHODS
            seeds = [('retrieval_seed', code)] + [
                (method, f'METHOD={method!r}\nOUTPUT_TOKENS={256 if getattr(args, "benchmark", "longlamp") == "longlamp" else 128}\n'+BASELINE)
                for method in METHODS]
            seed_results = []
            for name, source in seeds:
                fit_predictions = evaluate(source, 'seed_pool:'+name)
                held_predictions = evaluate(source, 'seed_selection:'+name, selection_rows)
                fit_summary = adapter.summarize_rows(fit_predictions, trace_limit=0)
                held_summary = adapter.summarize_rows(held_predictions, trace_limit=0)
                evo.write_jsonl(directory/(name+'_seed_fit.jsonl'), fit_predictions)
                evo.write_jsonl(directory/(name+'_seed_selection.jsonl'), held_predictions)
                seed_path = directory/('initial_'+name+'.py')
                evo.save_code(seed_path, source)
                if not fit_summary['errors']:
                    archive.append(dict(event='candidate_evaluated', candidate_id='initial_'+name,
                        code_path=str(seed_path), predictions_path=str(directory/(name+'_seed_fit.jsonl')),
                        iteration=0, operation='seed_pool', **{k:v for k,v in fit_summary.items() if k != 'worst_traces'}))
                seed_results.append(dict(name=name, fit=fit_summary, selection=held_summary))
            atomic_json(directory/'seed_pool_scores.json', seed_results)
            eligible = [i for i,r in enumerate(seed_results) if not r['fit']['errors'] and not r['selection']['errors']]
            if not eligible:
                raise RuntimeError('No seed passed both fit and historical selection evaluations')
            best = max(eligible, key=lambda i: (adapter.weighted_score(seed_results[i]['selection']),
                                              adapter.weighted_score(seed_results[i]['fit'])))
            code = seeds[best][1]
            seed_name = seeds[best][0]
            evo.write_jsonl(archive_path, archive)
        predictions = evaluate(code, 'seed')
        baseline = adapter.summarize_rows(predictions, trace_limit=len(rows))
        if baseline['errors']:
            evo.write_jsonl(directory/'seed_failed_predictions.jsonl', predictions)
            raise RuntimeError('Seed evaluation failed; do not optimize against a broken baseline')
        evo.save_code(directory / 'seed.py', code)
        evo.write_jsonl(directory / 'seed_predictions.jsonl', predictions)
        bank.add(adapter.build_failure_entries([], predictions, iteration=0, operation='seed',
                                              candidate_id='seed', max_entries=len(rows)))
        state = dict(protocol=protocol, benchmark=getattr(args, 'benchmark', 'longlamp'),
                     task=getattr(args, 'task', 'abstract_generation'), config_hash=config_hash, dataset_hash=dataset_hash,
                     user_id=user_id, baseline=baseline, summary=baseline, seed_name=seed_name,
                     completed_operations=0, exploration=None,
                     code_path=str(directory/'seed.py'),
                     predictions_path=str(directory/'seed_predictions.jsonl'))
        if selection_rows:
            selection_predictions = evaluate(code, 'seed_selection_recheck', selection_rows)
            selection_summary = adapter.summarize_rows(selection_predictions, trace_limit=0)
            if selection_summary['errors']:
                raise RuntimeError('Selected seed failed historical validation recheck')
            state['selection_summary'] = selection_summary
            state['selection_predictions_path'] = str(directory/'seed_selection_predictions.jsonl')
            evo.write_jsonl(directory/'seed_selection_predictions.jsonl', selection_predictions)
        if getattr(args, 'seed_pool', False):
            original = evo.load_jsonl(directory/(seed_name+'_seed_fit.jsonl'))
            differences = [p['sample_id'] for p,q in zip(original,predictions)
                           if p.get('prediction') != q.get('prediction')]
            atomic_json(directory/'seed_repeat_audit.json', dict(
                seed_name=seed_name, repeated_samples=len(predictions), changed_samples=differences,
                first_weighted_score=seed_results[best]['fit']['weighted_score'],
                repeat_weighted_score=baseline['weighted_score'],
                execution_audit=compare_executions(original, predictions),
                note='Fresh requests, no response cache; any changes indicate execution nondeterminism.'))
            if not compare_executions(original, predictions)['stable']:
                raise RuntimeError('Selected seed is not reproducible; inspect seed_repeat_audit.json before formal evolution')
            if selection_rows:
                held_audit = compare_executions(evo.load_jsonl(directory/(seed_name+'_seed_selection.jsonl')), selection_predictions)
                atomic_json(directory/'seed_selection_repeat_audit.json', held_audit)
                if not held_audit['stable']:
                    raise RuntimeError('Selected seed selection is not reproducible; inspect seed_selection_repeat_audit.json')
        atomic_json(state_path, state)

    # State and its completion event form a recoverable commit: the event is
    # embedded in the atomic state before publication to the monitor journal.
    if state.get('last_operation_event'):
        save_event(state['last_operation_event'])

    def read_branch(branch):
        return Path(branch['code_path']).read_text(), evo.load_jsonl(Path(branch['predictions_path'])), branch['summary']

    def publish(phase):
        summary, baseline = state['summary'], state['baseline']
        record = dict(user_id=user_id, phase=phase, samples=len(rows),
                      seed_name=state.get('seed_name', 'retrieval_seed'),
                      completed_operations=state['completed_operations'],
                      benchmark=getattr(args, 'benchmark', 'longlamp'),
                      task=getattr(args, 'task', 'abstract_generation'),
                      baseline_weighted_score_100=baseline['weighted_score_100'],
                      current_weighted_score_100=summary['weighted_score_100'],
                      gain_weighted_points=round(100*(summary['weighted_score']-baseline['weighted_score']), 3),
                      metric_protocol=summary.get('metric_protocol'),
                      objective_protocol=summary.get('objective_protocol'),
                      objective_weights=summary.get('objective_weights'),
                      harness=state['code_path'])
        for metric in adapter.metric_names:
            record[f'baseline_{metric}_100'] = baseline.get(f'mean_{metric}_100', 0.0)
            record[f'current_{metric}_100'] = summary.get(f'mean_{metric}_100', 0.0)
            record[f'gain_{metric}_points'] = round(100 * (
                summary.get(f'mean_{metric}', 0.0) - baseline.get(f'mean_{metric}', 0.0)), 3)
        # Preserve the legacy names consumed by existing LongLaMP monitors.
        if 'mean_rougeL' in summary:
            record.update(
                baseline_rougeL_100=baseline.get('mean_rougeL_100', 0.0),
                current_rougeL_100=summary.get('mean_rougeL_100', 0.0),
                gain_rougeL_points=round(100 * (summary.get('mean_rougeL', 0.0) - baseline.get('mean_rougeL', 0.0)), 3),
                rouge1_100=summary.get('mean_rouge1_100', 0.0),
                rougeL_100=summary.get('mean_rougeL_100', 0.0),
                bleu_100=summary.get('mean_bleu_100', 0.0),
                meteor_100=summary.get('mean_meteor_100', 0.0),
            )
        report(record)
    publish('baseline_ready')
    user_library = library + (
        '\nIndependent harness evolution for ONE isolated training user. '
        'No global user router or meta-policy. Specialize using this user\'s historical evidence. '
        'References are TRAINING diagnostics, never runtime inputs or hardcoded answers.\n' +
        adapter.task_context +
        '\nPreserve the current task title and all complete supplied keyword phrases in LLM inputs. '
        'Do not shorten a multiword keyword to its first word. '
        'Do not embed historical reference titles, headlines or abstracts in source literals; '
        'retrieve examples from row.profile instead. '
        'A disjoint historical selection gate checks generalization; its examples and scores are not provided here.')

    for op_index in range(state['completed_operations'], args.iterations):
        iteration = op_index + 1
        code, predictions, summary = read_branch(state)
        decision_path = directory / f'i{iteration:02d}_operation_choice.json'
        if decision_path.exists():
            decision = json.loads(decision_path.read_text())
        else:
            publish(f'choosing_operation:{iteration}')
            decision = evo.choose_operation(
                parent_code=code, iteration=iteration, current_summary=summary,
                failure_bank=bank, history=history, archive=archive,
                task_context=adapter.task_context, agent_api_url=args.agent_url,
                agent_api_model=agent_model, agent_timeout=agent_timeout,
                request_gate=gate, artifact_dir=directory/f'i{iteration:02d}_selection_requests')
            atomic_json(decision_path, decision)
        operation = decision['operation']
        if operation not in evo.OPERATION_NAMES:
            raise ValueError('Invalid persisted operation decision')
        operation_library = user_library + '\nChosen operation rationale: ' + decision['reason']
        incumbent = dict(code_path=state['code_path'], predictions_path=state['predictions_path'], summary=summary)
        if getattr(args, 'search_strategy', 'greedy') == 'archive_beam':
            # Every evaluated candidate remains a possible parent.  The
            # archive policy allocates proposal slots across the current best,
            # Pareto/novel nodes, and under-explored island representatives.
            groups = select_parent_branches(
                incumbent=incumbent, archive=archive, branch_count=args.branches,
                beam_width=getattr(args, 'beam_width', args.branches),
                archive_size=getattr(args, 'archive_size', 64),
                island_count=getattr(args, 'island_count', 4),
                iteration=iteration, operation=operation,
                incumbent_slots=getattr(args, 'incumbent_slots', 1),
            )
        else:
            groups = [(incumbent, args.branches)]
            exploratory = state.get('exploration')
            if operation == 'micro_repair' and exploratory and exploratory['code_path'] != incumbent['code_path']:
                # Legacy greedy mode: same total budget; repair both the
                # incumbent and one new hypothesis.
                explore_count = max(1, args.branches//2)
                groups = [(incumbent, args.branches-explore_count), (exploratory, explore_count)]
        op_dir = directory / f'i{iteration:02d}_{operation}'
        op_dir.mkdir(exist_ok=True)
        proposals_path = op_dir / 'proposals.json'
        publish(f'proposing:{iteration}:{operation}')
        if proposals_path.exists():
            proposals = json.loads(proposals_path.read_text())
        else:
            # Plan the full operation before assigning proposals to parents.
            # With archive-beam most parent groups have count=1; planning in
            # each group would otherwise silently fall back to a placeholder
            # hypothesis and erase the intended branch diversity.
            planned_hypotheses = None
            if getattr(args, 'search_strategy', 'greedy') == 'archive_beam':
                planned_hypotheses = evo.plan_hypotheses(
                    parent_code=code, operation=operation, iteration=iteration,
                    count=args.branches, current_summary=summary,
                    failure_bank=bank, history=history, archive=archive,
                    strategy_library=operation_library, agent_api_url=args.agent_url,
                    agent_api_model=agent_model, agent_timeout=agent_timeout,
                    agent_retries=2, agent_retry_wait=2, agent_max_tokens=None,
                    request_gate=gate, artifact_dir=op_dir/'agent_requests',
                    runtime_model=qa_model, runtime_context=qa_context,
                    task_context=adapter.task_context,
                    objective_description=adapter.objective_description,
                    evaluation_boundary=adapter.evaluation_boundary)
            proposals = []
            proposal_offset = 0
            for branch_parent, count in groups:
                if not count:
                    continue
                pcode, prows, psummary = read_branch(branch_parent)
                proposal_kwargs = dict(
                    parent_code=pcode, operation=operation, iteration=iteration, count=count,
                    current_summary=psummary, failure_bank=bank, history=history, archive=archive,
                    strategy_library=operation_library, agent_api_url=args.agent_url,
                    agent_api_model=agent_model, agent_timeout=agent_timeout, agent_retries=2,
                    agent_retry_wait=2, agent_concurrency=count, agent_max_tokens=None,
                    request_gate=gate, artifact_dir=op_dir/'agent_requests',
                    runtime_model=qa_model, runtime_context=qa_context,
                    task_context=adapter.task_context,
                    objective_description=adapter.objective_description,
                    evaluation_boundary=adapter.evaluation_boundary)
                if planned_hypotheses is not None:
                    proposal_kwargs['planned_hypotheses'] = planned_hypotheses[
                        proposal_offset:proposal_offset + count]
                group = evo.propose_candidates(**proposal_kwargs)
                for proposal in group:
                    proposal['source_parent'] = branch_parent
                    proposals.append(proposal)
                proposal_offset += count
            if proposals and all(p.get('failure_kind') == 'service_request' for p in proposals):
                raise RuntimeError('All evolver requests failed; operation not advanced: ' + str(proposals[0].get('error')))
            atomic_json(proposals_path, proposals)

        if len(proposals) > args.branches:
            raise ValueError('Persisted/generated proposal count exceeds round budget')

        evaluated = []
        seen = {evo.behavior_fingerprint(code)}
        for branch, proposal in enumerate(proposals):
            cid = f'i{iteration:02d}_{operation}_b{branch}'
            source_parent = proposal.get('source_parent', incumbent)
            event = dict(user_id=user_id, iteration=iteration, operation=operation, candidate_id=cid,
                         accepted=False, hypothesis=proposal.get('hypothesis', ''),
                         exploration_parent=source_parent['code_path'],
                         parent_code_path=source_parent['code_path'],
                         parent_candidate_id=source_parent.get('candidate_id', 'incumbent'),
                         island_id=source_parent.get('island_id'))
            if proposal.get('valid'):
                leak = evo.reference_literal_leak(proposal['code'], rows)
                if leak:
                    proposal = dict(proposal, valid=False, error=leak)
            if proposal.get('valid'):
                fingerprint = evo.behavior_fingerprint(proposal['code'])
                if fingerprint in seen:
                    proposal = dict(proposal, valid=False, error='duplicate AST within operation or incumbent')
                seen.add(fingerprint)
            if not proposal.get('valid'):
                event.update(event='invalid_code', error=proposal.get('error'))
                bank.add([dict(event, failure_types=['invalid_code'])])
                atomic_json(directory/f'{cid}_invalid.json', proposal)
                save_event(event)
                continue
            path = directory/f'{cid}.py'
            pred_path, cached_path = directory/f'{cid}_predictions.jsonl', directory/f'{cid}_evaluation.json'
            evo.save_code(path, proposal['code'])
            _, parent_rows, parent_summary = read_branch(source_parent)

            # A cheap isolated execution catches interface/type mistakes before
            # spending eight QA evaluations on a candidate that cannot run.
            # Service-side errors are allowed through to the full evaluation so
            # a transient vLLM failure is not mistaken for a bad harness.
            smoke_rows = rows[:1]
            smoke_predictions = evaluate(proposal['code'], cid + ':smoke', smoke_rows)
            smoke_summary = adapter.summarize_rows(smoke_predictions, trace_limit=len(smoke_rows))
            evo.write_jsonl(directory/(cid+'_smoke_predictions.jsonl'), smoke_predictions)
            atomic_json(directory/(cid+'_smoke_evaluation.json'), smoke_summary)
            smoke_errors = [str(item.get('error', '')) for item in smoke_predictions if item.get('error')]
            service_markers = (
                'vllm request failed', 'connection refused', 'connection reset',
                'timed out', 'urlopen error', 'http error', '502', '503', '504',
            )
            harness_smoke_errors = [
                error for error in smoke_errors
                if not any(marker in error.lower() for marker in service_markers)
            ]
            if harness_smoke_errors:
                error = '; '.join(harness_smoke_errors)[:1000]
                event.update(
                    event='runtime_smoke_failed',
                    error=error,
                    smoke_errors=smoke_errors,
                    smoke_weighted_score_100=smoke_summary['weighted_score_100'],
                )
                bank.add(adapter.build_failure_entries(
                    parent_rows, smoke_predictions, iteration=iteration,
                    operation=operation, candidate_id=cid, max_entries=len(smoke_rows)))
                atomic_json(directory/(cid+'_smoke_failed.json'), {
                    **proposal, 'error': error, 'smoke_errors': smoke_errors,
                })
                save_event(event)
                continue
            publish(f'evaluating:{cid}')
            if cached_path.exists() and pred_path.exists():
                child_summary = json.loads(cached_path.read_text())
                child_rows = evo.load_jsonl(pred_path)
            else:
                child_rows = evaluate(proposal['code'], cid)
                child_summary = adapter.summarize_rows(child_rows, trace_limit=len(rows))
                evo.write_jsonl(pred_path, child_rows)
                atomic_json(cached_path, child_summary)
            bank.add(adapter.build_failure_entries(parent_rows, child_rows, iteration=iteration,
                                                  operation=operation, candidate_id=cid, max_entries=len(rows)))
            event.update(event='candidate_evaluated', code_path=str(path), predictions_path=str(pred_path),
                         parent_weighted_score=parent_summary['weighted_score'],
                         child_weighted_score=child_summary['weighted_score'],
                         delta_weighted_score=child_summary['weighted_score']-parent_summary['weighted_score'],
                         delta_weighted_points=round(100*(child_summary['weighted_score']-parent_summary['weighted_score']), 3),
                         **{k:v for k,v in child_summary.items() if k != 'worst_traces'})
            for metric in adapter.metric_names:
                event[f'parent_{metric}'] = parent_summary.get(f'mean_{metric}', 0.0)
                event[f'child_{metric}'] = child_summary.get(f'mean_{metric}', 0.0)
                event[f'delta_{metric}'] = child_summary.get(f'mean_{metric}', 0.0) - parent_summary.get(f'mean_{metric}', 0.0)
            archive[:] = [r for r in archive if r['candidate_id'] != cid]
            archive.append(event.copy())
            evo.write_jsonl(archive_path, archive)
            save_event(event)
            evaluated.append(dict(code_path=str(path), predictions_path=str(pred_path),
                                  summary=child_summary, candidate_id=cid))

        # Validation examples/predictions never enter prompts, archives or the
        # failure bank. Selection is adaptive, not an untouched test estimate.
        for candidate in evaluated:
            if selection_rows and not candidate['summary']['errors']:
                ccode, _, _ = read_branch(candidate)
                held = evaluate(ccode, candidate['candidate_id']+':history_selection', selection_rows)
                candidate['selection_summary'] = adapter.summarize_rows(held, trace_limit=0)
                evo.write_jsonl(directory/(candidate['candidate_id']+'_selection_predictions.jsonl'), held)
        # Best-observed incumbent and exploratory branch are separate states.
        accepted_id = None
        ranked = sorted(evaluated, key=lambda b: (
            b['summary']['errors'],
            b.get('selection_summary', {}).get('errors', 0),
            -adapter.weighted_score(b.get('selection_summary', b['summary'])),
            tuple(-value for value in adapter.score_key(b['summary'])),
        ))
        for candidate in ranked:
            if not historical_improvement(adapter, candidate['summary'], summary,
                    candidate.get('selection_summary'), state.get('selection_summary')):
                atomic_json(directory/(candidate['candidate_id']+'_admission.json'), dict(
                    accepted=False, reason='no_error_free_hidden_selection_gain_without_fit_regression'))
                continue
            publish('confirming:' + candidate['candidate_id'])
            ccode, _, _ = read_branch(candidate)
            cpath = directory/(candidate['candidate_id']+'_confirmation.json')
            if cpath.exists():
                confirmation = json.loads(cpath.read_text())
            else:
                confirmed = evaluate(ccode, candidate['candidate_id']+':confirm')
                parent_repeat = evaluate(code, candidate['candidate_id']+':parent_recheck')
                confirmation = dict(candidate=adapter.summarize_rows(confirmed, trace_limit=len(rows)),
                                    parent=adapter.summarize_rows(parent_repeat, trace_limit=len(rows)))
                confirmation['execution_audit'] = {
                    'candidate_fit': compare_executions(evo.load_jsonl(Path(candidate['predictions_path'])), confirmed),
                    'parent_fit': compare_executions(evo.load_jsonl(Path(state['predictions_path'])), parent_repeat),
                }
                if selection_rows:
                    held_child = evaluate(ccode, candidate['candidate_id']+':selection_confirm', selection_rows)
                    held_parent = evaluate(code, candidate['candidate_id']+':selection_parent_recheck', selection_rows)
                    confirmation['selection_candidate'] = adapter.summarize_rows(held_child, trace_limit=0)
                    confirmation['selection_parent'] = adapter.summarize_rows(held_parent, trace_limit=0)
                    confirmation['execution_audit']['candidate_selection'] = compare_executions(
                        evo.load_jsonl(directory/(candidate['candidate_id']+'_selection_predictions.jsonl')), held_child)
                    confirmation['execution_audit']['parent_selection'] = compare_executions(
                        evo.load_jsonl(Path(state['selection_predictions_path'])), held_parent)
                    evo.write_jsonl(directory/(candidate['candidate_id']+'_selection_confirmed.jsonl'), held_child)
                    evo.write_jsonl(directory/(candidate['candidate_id']+'_selection_parent.jsonl'), held_parent)
                evo.write_jsonl(directory/(candidate['candidate_id']+'_confirmed_predictions.jsonl'), confirmed)
                evo.write_jsonl(directory/(candidate['candidate_id']+'_parent_recheck.jsonl'), parent_repeat)
                atomic_json(cpath, confirmation)
            repeat_summary = confirmation['candidate']
            # Same training data, fresh executions; not a claim of statistical significance.
            accepted = historical_improvement(adapter, repeat_summary, confirmation['parent'],
                confirmation.get('selection_candidate'), confirmation.get('selection_parent')) and historical_improvement(
                adapter, repeat_summary, summary, confirmation.get('selection_candidate'), state.get('selection_summary'))
            stable = all(a['stable'] for a in confirmation.get('execution_audit', {}).values())
            accepted = accepted and stable
            atomic_json(directory/(candidate['candidate_id']+'_admission.json'), dict(
                accepted=accepted, reason='accepted' if accepted else (
                    'execution_instability' if not stable else 'confirmation_failed'),
                execution_audit=confirmation.get('execution_audit', {})))
            confirmation_event = dict(event='confirmation', user_id=user_id, iteration=iteration, operation=operation,
                                      candidate_id=candidate['candidate_id'], accepted=accepted,
                                      execution_stable=stable,
                                      child_weighted_score=repeat_summary['weighted_score'],
                                      parent_weighted_score=confirmation['parent']['weighted_score'],
                                      delta_weighted_points=round(100*(repeat_summary['weighted_score']-
                                                                       confirmation['parent']['weighted_score']), 3))
            for metric in adapter.metric_names:
                confirmation_event[f'child_{metric}'] = repeat_summary.get(f'mean_{metric}', 0.0)
                confirmation_event[f'parent_{metric}'] = confirmation['parent'].get(f'mean_{metric}', 0.0)
            save_event(confirmation_event)
            if accepted:
                if selection_rows:
                    state['selection_summary'] = confirmation['selection_candidate']
                    state['selection_predictions_path'] = str(directory/(candidate['candidate_id']+'_selection_confirmed.jsonl'))
                accepted_id = candidate['candidate_id']
                state.update(code_path=candidate['code_path'],
                             predictions_path=str(directory/(accepted_id+'_confirmed_predictions.jsonl')),
                             summary=repeat_summary)
                break
        for item in archive:
            if item['candidate_id'] == accepted_id:
                item['accepted'] = True
        evo.write_jsonl(archive_path, archive)
        if operation == 'macro_strategy':
            # Retain an alternative for a later round if repair is chosen.
            alternatives = [b for b in ranked if b['candidate_id'] != accepted_id]
            state['exploration'] = alternatives[0] if alternatives else None
        else:
            state['exploration'] = None
        state['completed_operations'] = op_index+1
        operation_event = dict(user_id=user_id, event='operation_complete', iteration=iteration, operation=operation,
                              accepted_candidate=accepted_id,
                              weighted_score_100=state['summary']['weighted_score_100'],
                              exploration_parent=(state.get('exploration') or {}).get('code_path'),
                              code_path=state['code_path'], operation_reason=decision['reason'])
        for metric in adapter.metric_names:
            operation_event[f'{metric}_100'] = state['summary'].get(f'mean_{metric}_100', 0.0)
        if 'mean_rougeL_100' in state['summary']:
            operation_event['rougeL_100'] = state['summary']['mean_rougeL_100']
        state['last_operation_event'] = operation_event
        atomic_json(state_path, state)
        evo.save_code(directory/'current_harness.py', Path(state['code_path']).read_text())
        save_event(operation_event)
        publish('operation_complete')
    publish('complete')
    return state


class LimitedQA(evo.VLLMOpenAIClient):
    def __init__(self, *args, request_limit=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_gate = threading.BoundedSemaphore(request_limit)
    def _request(self, prompt, payload):
        with self.request_gate:
            return super()._request(prompt, payload)


def wait_for_evolver(url, output_dir, model_id):
    """Keep a recoverable launch waiting rather than fail every queued user."""
    while True:
        try:
            with urllib.request.urlopen(url.rstrip('/') + '/models', timeout=10) as response:
                models = json.load(response).get('data', [])
            if not any(m.get('id') == model_id for m in models):
                raise RuntimeError(f'Required agent model {model_id} is not served')
            atomic_json(output_dir / 'service_status.json', {
                'phase':'evolver_available', 'url':url, 'model': model_id, 'time':time.time()})
            return
        except Exception as error:
            status = {'phase':'waiting_for_evolver', 'url':url, 'time':time.time(),
                      'error':f'{type(error).__name__}: {error}'}
            atomic_json(output_dir / 'service_status.json', status)
            print(json.dumps(status), flush=True)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=('longlamp', 'lamp'), default='longlamp')
    parser.add_argument('--task', type=int, choices=(2, 3, 4, 5),
                        help='LaMP task; required when --benchmark lamp')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='Disjoint historical selection tasks, never official test tasks')
    parser.add_argument('--seed-pool', action='store_true', help='Select among retrieval, full context, RAG, PAG and CoT using history only')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of rounds; one agent-chosen operation per round')
    parser.add_argument('--branches', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--search-strategy', choices=('greedy', 'archive_beam'), default='greedy')
    parser.add_argument('--beam-width', type=int, default=4)
    parser.add_argument('--archive-size', type=int, default=64)
    parser.add_argument('--island-count', type=int, default=4)
    parser.add_argument('--incumbent-slots', type=int, default=1,
                        help='Reserve this many candidates for incumbent; remaining slots explore archive')
    parser.add_argument('--user-workers', type=int, default=4)
    parser.add_argument('--agent-concurrency', type=int, default=8)
    parser.add_argument('--qa-concurrency', type=int, default=32,
                        help='Shared in-flight Answer Model request cap for this configuration')
    parser.add_argument('--max-users', type=int)
    parser.add_argument('--sample-timeout', type=float, default=300)
    parser.add_argument('--user-retries', type=int, default=2)
    parser.add_argument('--qa-url', default='http://gpu01:8000/v1')
    parser.add_argument('--qa-model', default='Qwen2.5-7B-Instruct')
    parser.add_argument('--agent-url', default='http://gpu02:18012/v1')
    parser.add_argument('--agent-model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--agent-timeout', type=float, default=1200)
    parser.add_argument('--embedding-url', default='http://gpu01:18013/v1')
    parser.add_argument('--embedding-backend', choices=('vllm', 'cpu'), default='vllm')
    args = parser.parse_args()
    if args.agent_model != 'Qwen/Qwen3.8-27B':
        parser.error('Code Agent must be Qwen/Qwen3.8-27B')
    if not 1 <= args.incumbent_slots <= args.branches:
        parser.error('incumbent-slots must be between 1 and branches')
    if args.benchmark == 'lamp' and args.task is None:
        parser.error('--task is required when --benchmark lamp')
    if args.benchmark == 'longlamp':
        args.task = 'abstract_generation'
    adapter = adapter_for(args)
    if min(args.iterations, args.branches, args.beam_width, args.archive_size,
           args.island_count, args.user_workers, args.agent_concurrency, args.qa_concurrency) < 1:
        parser.error('iterations/branches/search widths/concurrency must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_lock = (args.output_dir/'runner.lock').open('a+')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(args.output_dir/'runner.json', {'pid': os.getpid(), 'started': time.time()})
    metric_config = adapter.preflight()
    root = Path(__file__).resolve().parents[1]
    source_files = [root/'scripts'/name for name in (
        'evolve_per_user.py', 'code_evolution.py', 'blackbox_harness.py', 'longlamp_rsi.py',
        'isolated_runtime.py', 'harness_worker.py', 'embedding_client.py', 'evolution_metrics.py',
        'search_policy.py', 'lamp_tasks.py', 'evaluate_lamp_test.py',
        'prepare_lamp_user_tta.sh', 'run_lamp_evo.sh',
        'run_per_user_evo.sh', 'paired_suite.py', 'profile_protocol.py')] + [root/'docs'/'BLACKBOX_STRATEGY_LIBRARY.md']
    args.source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()
    previous_config = args.output_dir/'run_config.json'
    if previous_config.exists() and json.loads(previous_config.read_text()).get('source_hash') != args.source_hash:
        raise RuntimeError('Source changed since launch; use a new run directory, do not mix protocols')
    snapshot = args.output_dir/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for source in source_files:
        shutil.copy2(source, snapshot/source.name)
    groups = user_groups(evo.load_jsonl(args.train))
    selection_groups = user_groups(evo.load_jsonl(args.selection))
    if set(groups) != set(selection_groups):
        raise ValueError('Fit/selection user cohorts must match exactly')
    from scripts.profile_protocol import history_key
    for user, held in selection_groups.items():
        if not held or any(r.get('source_split') != 'profile_selection' for r in held):
            raise ValueError('Historical selection tasks required; official tests are forbidden')
        protected = {r.get('historical_key') for r in held}
        if None in protected or len(protected) != len(held):
            raise ValueError('Historical selection keys missing or duplicated')
        if any(r.get('historical_key') in protected or
               any(history_key(p) in protected for p in r['profile']) for r in groups[user]):
            raise ValueError('Selection examples leaked into fitting rows or profiles')
    for user, items in groups.items():
        if len({r['sample_id'] for r in items}) != len(items) or any(not r.get('target') or not r.get('input') for r in items):
            raise ValueError('Duplicate samples or missing train input/reference for ' + user)
    users = sorted(groups)
    if args.max_users:
        users = users[:args.max_users]
    with urllib.request.urlopen(args.qa_url.rstrip('/')+'/models', timeout=15) as response:
        qa_models = json.load(response)['data']
    qa_model = next((m for m in qa_models if m['id'] == args.qa_model), None)
    if qa_model is None:
        raise RuntimeError(f'Required frozen QA model {args.qa_model} is not served')
    if qa_model.get('max_model_len', 8192) < 8192:
        raise RuntimeError('QA context is smaller than the advertised harness contract')
    args.qa_context = int(qa_model.get('max_model_len', 8192))
    qa_template = {'enable_thinking': False} if 'qwen3' in args.qa_model.lower() else None
    qa = LimitedQA(args.qa_url, args.qa_model, concurrency=args.qa_concurrency, request_limit=args.qa_concurrency,
                   timeout=120, retries=1, chat_template_kwargs=qa_template)
    qa.embedding_client = EmbeddingClient(args.embedding_url,
        truncate_prompt_tokens=2048 if args.embedding_backend == 'vllm' else None)
    if len(qa.embedding_client.embed(['per-user preflight'])[0]) != 1024:
        raise RuntimeError('Embedding dimension differs from runtime contract')
    packages = {name: importlib.import_module(name).__version__ for name in ('numpy','scipy','sklearn','networkx')}
    library = tools_reference(packages)
    gate = threading.BoundedSemaphore(args.agent_concurrency)
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(unit='one_user_one_harness', users=len(users), samples=sum(len(groups[u]) for u in users),
                  validation_used=True, validation_source='disjoint_historical_profile', test_used=False, packages=packages,
                  protocol=protocol_for(args), metric_config=metric_config,
                  search_strategy=args.search_strategy, beam_width=args.beam_width,
                  archive_size=args.archive_size, island_count=args.island_count,
                  evolver_output_limit=None, evolver_min_tokens=None)
    atomic_json(args.output_dir / 'run_config.json', config)
    records_path = args.output_dir/'user_scores.json'
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    lock = threading.Lock()
    def report(record):
        with lock:
            record['time'] = time.time()
            records[record['user_id']] = {**records.get(record['user_id'], {}), **record}
            if record['phase'] not in ('failed', 'retry_wait'):
                records[record['user_id']].pop('error', None)
            atomic_json(args.output_dir / 'user_scores.json', records)
            gains = [r['gain_weighted_points'] for r in records.values() if 'gain_weighted_points' in r]
            complete = [r for r in records.values() if r['phase'] == 'complete']
            atomic_json(args.output_dir/'progress.json', dict(users_total=len(users),
                users_with_baseline=len(gains), users_complete=len(complete),
                users_failed=sum(r['phase']=='failed' for r in records.values()),
                users_improved=sum(g>0 for g in gains), users_no_gain=sum(g<=0 for g in gains),
                median_gain_points=statistics.median(gains) if gains else None,
                min_gain_points=min(gains) if gains else None))
            print(json.dumps(record, ensure_ascii=False), flush=True)
    print(json.dumps({'event':'start', **config}, ensure_ascii=False), flush=True)
    wait_for_evolver(args.agent_url, args.output_dir, args.agent_model)
    failures = 0
    def run_user(user):
        for attempt in range(args.user_retries+1):
            try:
                return evolve_user(user, groups[user], args, qa, gate, library, report, adapter=adapter,
                                   selection_rows=selection_groups[user])
            except Exception as error:
                if attempt == args.user_retries:
                    raise
                report({'user_id': user, 'phase': 'retry_wait', 'attempt': attempt+1,
                        'error': type(error).__name__ + ': ' + str(error)})
                time.sleep(10)
                wait_for_evolver(args.agent_url, args.output_dir, args.agent_model)
    with ThreadPoolExecutor(max_workers=args.user_workers) as pool:
        futures = {pool.submit(run_user, u):u for u in users}
        for future in as_completed(futures):
            user_id = futures[future]
            try:
                future.result()
            except Exception as error:
                failures += 1
                report({'user_id':user_id, 'phase':'failed', 'error':f'{type(error).__name__}: {error}'})
    print(json.dumps({'event':'finished', 'users':len(users), 'failed_users':failures}), flush=True)
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
