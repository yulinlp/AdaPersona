"""Reproducible paired baseline + per-user RSI suite, using existing services.

All test artifacts live outside evolution directories. No test score is used
to select users, candidates, stopping points, or operations.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from scripts import code_evolution as evo
from scripts.evolve_per_user import LongLaMPAdapter, LimitedQA, atomic_json
from scripts.lamp_tasks import LampTaskAdapter
from scripts.embedding_client import EmbeddingClient

MODELS = {
    'qwen25': ('http://gpu01:8000/v1', 'Qwen2.5-7B-Instruct'),
    'qwen38': ('http://gpu02:18012/v1', 'Qwen/Qwen3.8-27B'),
}
METHODS = ('full_context', 'rag', 'pag', 'cot')

BASELINE = r'''
import math
import re
from collections import Counter

def tokens(text):
    return re.findall(r"\w+", str(text).lower())

def render(item):
    return "\n".join(str(k) + ": " + str(v) for k,v in item.items() if k != "id")

def run(row, qa):
    history = list(row.get('profile') or [])
    query = str(row['input'])
    if len(query) > 12000:
        raise ValueError('Current input exceeds baseline safety budget; do not silently truncate task')
    docs = [render(item) for item in history]
    if METHOD == 'rag':
        ts = [tokens(d) for d in docs]
        df = Counter(t for d in ts for t in set(d))
        avg = sum(map(len, ts)) / max(1, len(ts))
        q = set(tokens(query))
        scored = []
        for i, d in enumerate(ts):
            tf = Counter(d)
            score = sum(math.log(1+(len(ts)-df[t]+0.5)/(df[t]+0.5)) *
                        tf[t]*2.2/(tf[t]+1.2*(0.25+0.75*len(d)/max(1,avg)))
                        for t in q if tf[t])
            scored.append((-score, i))
        docs = [docs[i] for _,i in sorted(scored)[:2]]
    # Identical conservative character budget across answer-model conditions.
    # Full context means all history fitting this budget, not unbounded history.
    context = '\n\n'.join(docs)[:max(0, 18000-len(query))]
    if METHOD == 'pag':
        context = qa.generate('Infer a compact user profile from these historical examples. '
            'Describe evidence-supported preferences, task conventions and style; '
            'do not invent facts.\n' + context, max_tokens=256)
    prompt = ('Use the historical evidence to perform the current task. '
              'Return only the requested answer, category, rating, title, or abstract.\n'
              'USER HISTORY / PROFILE:\n' + context + '\nCURRENT TASK:\n' + query)
    if METHOD == 'cot':
        plan = qa.generate(prompt + '\nFirst make a brief evidence-grounded plan, not the final answer.',
                           max_tokens=192)
        prompt += '\nPLAN:\n' + plan + '\nNow output only the final answer.'
    return qa.generate(prompt, max_tokens=OUTPUT_TOKENS)
'''

def adapter_for(spec):
    return LongLaMPAdapter() if spec['benchmark'] == 'longlamp' else LampTaskAdapter(spec['task'])

def prepare(root, users):
    specs = []
    sources = [('longlamp', None, Path('data/experiments/longlamp_abstract_user_rsi_tta'))]
    sources += [('lamp', t, Path(f'data/experiments/lamp_user_rsi/lamp_{t}')) for t in (2,3,4,5)]
    for benchmark, task, source in sources:
        name = 'longlamp_abstract' if benchmark == 'longlamp' else f'lamp_{task}'
        cohort = root / 'cohorts' / name
        cohort.mkdir(parents=True, exist_ok=True)
        # File-order cohort is predeclared, independent of any measured scores.
        test_rows = []
        with (source/'test.jsonl').open() as handle:
            for line in handle:
                if line.strip():
                    test_rows.append(json.loads(line))
                if len(test_rows) >= users:
                    break
        ids = {str(r['user_id']) for r in test_rows}
        train = []
        with (source/'profile_adaptation.jsonl').open() as handle:
            for line in handle:
                # Read one row at a time; do not load multi-GB full datasets.
                row = json.loads(line)
                if str(row['user_id']) in ids:
                    train.append(row)
        if len(ids) != users or ids != {str(r['user_id']) for r in train}:
            raise ValueError(f'Incomplete or duplicate cohort: {name}')
        evo.write_jsonl(cohort/'test.jsonl', test_rows)
        evo.write_jsonl(cohort/'profile_adaptation.jsonl', train)
        atomic_json(cohort/'manifest.json', dict(users=sorted(ids), selection='first N in source file; no scores used',
            source=str(source), adaptation_rows=len(train), test_rows=len(test_rows),
            train_sha256=hashlib.sha256((cohort/'profile_adaptation.jsonl').read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256((cohort/'test.jsonl').read_bytes()).hexdigest()))
        for model in MODELS:
            specs.append(dict(name=f'{name}_{model}', benchmark=benchmark, task=task,
                              model=model, cohort=str(cohort), users=sorted(ids)))
    return specs

def run_baselines(spec, root):
    adapter = adapter_for(spec)
    adapter.preflight()
    url, model = MODELS[spec['model']]
    qa = LimitedQA(url, model, concurrency=2, request_limit=2, timeout=180, retries=2,
                   chat_template_kwargs={'enable_thinking': False})
    qa.embedding_client = EmbeddingClient('http://gpu01:18013/v1', truncate_prompt_tokens=2048)
    test = evo.load_jsonl(Path(spec['cohort'])/'test.jsonl')
    destination = root/'baseline'/spec['name']
    destination.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        code = f'METHOD = {method!r}\nOUTPUT_TOKENS = {256 if spec["benchmark"] == "longlamp" else 128}\n' + BASELINE
        evo.save_code(destination/(method+'.py'), code)
        path = destination/(method+'_predictions.jsonl')
        records = evo.load_jsonl(path) if path.exists() else []
        for row in test:
            if any(r['user_id'] == row['user_id'] and not r.get('error') for r in records):
                continue
            prediction = evo.evaluate_code(code, [row], qa, qa_max_calls=8,
                qa_concurrency=1, sample_timeout=420, label=f'{spec["name"]}:{method}')
            records = [r for r in records if r['user_id'] != row['user_id']] + prediction
            evo.write_jsonl(path, records)
            atomic_json(destination/(method+'_summary.json'), adapter.summarize_rows(records, trace_limit=0))
        if any(r.get('error') for r in records) or len(records) != len(test):
            raise RuntimeError(f'Incomplete baseline {spec["name"]}:{method}; inspect predictions')

def monitor_command(spec, root):
    command = [sys.executable, '-m', 'scripts.monitor_lamp_test', '--benchmark', spec['benchmark'],
        '--run-dir', str(root/'rsi'/spec['name']), '--test', str(Path(spec['cohort'])/'test.jsonl'),
        '--output-dir', str(root/'test_monitor'/spec['name']), '--once', '--user-workers', '1', '--qa-concurrency', '2']
    if spec['task'] is not None:
        command += ['--task', str(spec['task'])]
    return command

def run_rsi(spec, root, iterations, branches):
    run = root/'rsi'/spec['name']
    url, model = MODELS[spec['model']]
    command = [sys.executable, '-u', '-m', 'scripts.evolve_per_user',
        '--benchmark', spec['benchmark'], '--train', str(Path(spec['cohort'])/'profile_adaptation.jsonl'),
        '--output-dir', str(run), '--iterations', str(iterations), '--branches', str(branches),
        '--search-strategy', 'archive_beam', '--beam-width', '4', '--archive-size', '64', '--island-count', '4',
        '--agent-url', MODELS['qwen38'][0], '--agent-model', MODELS['qwen38'][1],
        '--qa-url', url, '--qa-model', model, '--user-workers', '1', '--agent-concurrency', '1']
    if spec['task'] is not None:
        command += ['--task', str(spec['task'])]
    with (root/'logs'/(spec['name']+'_rsi.log')).open('a') as log, (root/'logs'/(spec['name']+'_monitor.log')).open('a') as monitorlog:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=root/'code_snapshot')
        atomic_json(root/'status'/(spec['name']+'.json'), dict(phase='rsi_running', pid=process.pid, command=command))
        try:
            while process.poll() is None:
                if (run/'run_config.json').exists():
                    # Every committed round is discovered; failed monitor calls
                    # retry at the next poll without exposing test feedback.
                    subprocess.run(monitor_command(spec, root), stdout=monitorlog, stderr=subprocess.STDOUT,
                                   cwd=root/'code_snapshot')
                time.sleep(15)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        if process.returncode:
            raise RuntimeError(f'RSI exited {process.returncode}')
        result = subprocess.run(monitor_command(spec, root), stdout=monitorlog, stderr=subprocess.STDOUT,
                                cwd=root/'code_snapshot')
        if result.returncode:
            raise RuntimeError('Final test monitor failed')
    verify_complete(spec, root, iterations)

def verify_complete(spec, root, iterations):
    from scripts.evolve_per_user import user_key
    scores = evo.load_jsonl(root/'test_monitor'/spec['name']/'per_round_scores.jsonl')
    for user in spec['users']:
        state = json.loads((root/'rsi'/spec['name']/'users'/user_key(user)/'state.json').read_text())
        if state['completed_operations'] != iterations:
            raise RuntimeError('Incomplete RSI rounds')
        done = {r['iteration'] for r in scores if r['user_id'] == user and not r['errors']}
        if done != set(range(iterations+1)):
            raise RuntimeError(f'Missing/error test rounds for {user}: {sorted(done)}')
        for method in METHODS:
            rows = evo.load_jsonl(root/'baseline'/spec['name']/(method+'_predictions.jsonl'))
            if not any(r['user_id'] == user and not r.get('error') for r in rows):
                raise RuntimeError('Missing paired baseline')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--users', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--branches', type=int, choices=(1,2,3), default=3)
    parser.add_argument('--only', help='Optional exact configuration name for smoke tests')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for folder in ('logs', 'status'):
        (root/folder).mkdir(exist_ok=True)
    import fcntl
    lock = (root/'suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for url, model in MODELS.values():
        with urllib.request.urlopen(url+'/models', timeout=15) as response:
            if model not in [m['id'] for m in json.load(response)['data']]:
                raise RuntimeError(f'Model not served: {model}')
    manifest_path = root/'suite.json'
    source_root = Path(__file__).resolve().parents[1]
    source_files = sorted((source_root/'scripts').glob('*.py')) + sorted((source_root/'scripts').glob('*.sh'))
    source_files += sorted((source_root/'docs').glob('*.md')) + [source_root/'README.md']
    source_hash = hashlib.sha256(b''.join(str(p.relative_to(source_root)).encode()+p.read_bytes() for p in source_files)).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('source_hash') != source_hash:
            raise ValueError('Suite source changed; use a fresh experiment directory')
        if manifest['iterations'] != args.iterations or manifest['branches'] != args.branches or manifest['users_per_task'] != args.users:
            raise ValueError('Cannot change suite protocol on resume')
        specs = manifest['configurations']
    else:
        # Execute workers/monitors from an immutable source copy, so editing
        # the workspace later cannot silently change an in-flight experiment.
        for path in source_files:
            target = root/'code_snapshot'/path.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        # Metrics locate local resources relative to their source root.
        resource = root/'code_snapshot'/'data'
        resource.symlink_to(source_root/'data', target_is_directory=True)
        specs = prepare(root, args.users)
        if args.only:
            specs = [s for s in specs if s['name'] == args.only]
        if not specs:
            raise ValueError('No matching configuration')
        atomic_json(manifest_path, dict(configurations=specs, iterations=args.iterations, branches=args.branches,
            users_per_task=args.users, models=MODELS, methods=METHODS, created=time.time(), source_hash=source_hash,
            baseline_definition='local black-box implementations, not paper-exact reproductions'))
    def execute(spec):
        status = root/'status'/(spec['name']+'.json')
        try:
            atomic_json(status, dict(phase='baseline_running'))
            run_baselines(spec, root)
            run_rsi(spec, root, args.iterations, args.branches)
            atomic_json(status, dict(phase='complete', paired_users=spec['users']))
            return True
        except Exception as error:
            atomic_json(status, dict(phase='failed', error=type(error).__name__+': '+str(error)))
            return False
    for spec in specs:
        atomic_json(root/'status'/(spec['name']+'.json'), dict(phase='queued'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(execute, specs))
    atomic_json(root/'completion.json', dict(complete=all(results), settings=len(results), failed=results.count(False)))
    if not all(results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
