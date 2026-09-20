"""Fresh historical-only repeat audit; never caches or consults test scores."""
import argparse
import hashlib
from pathlib import Path

from scripts import code_evolution as evo
from scripts.execution_audit import compare_executions
from scripts.evolve_per_user import atomic_json
from scripts.paired_suite import MODELS
from scripts.embedding_client import EmbeddingClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--items', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2],
                        help='Test sequential and concurrent execution shapes on identical tasks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.items < 1 or args.repeats < 2 or min(args.concurrency) < 1:
        parser.error('Need positive items and at least two fresh executions')
    model_key = args.configuration.rsplit('_', 1)[-1]
    url, model = MODELS[model_key]
    qa = evo.VLLMOpenAIClient(url, model, concurrency=2, timeout=180, retries=0,
                              chat_template_kwargs={'enable_thinking': False})
    qa.embedding_client = EmbeddingClient('http://gpu01:18013/v1', truncate_prompt_tokens=2048)
    report = dict(model=model, endpoint=url, configuration=args.configuration, concurrency=args.concurrency,
                  repeats=args.repeats, users=[], stable=True, response_cache=False)
    directories = sorted((args.run_dir/'rsi'/args.configuration/'users').glob('*'))
    if not directories:
        raise ValueError('No historical user executions found')
    for directory in directories:
        rows = evo.load_jsonl(directory/'seed_predictions.jsonl')[:args.items]
        if not rows or any(r.get('source_split') != 'profile_fit' for r in rows):
            raise ValueError('Audit requires historical fitting tasks; official test tasks forbidden')
        code = (directory/'seed.py').read_text()
        executions = [evo.evaluate_code(code, rows, qa, qa_max_calls=8, qa_concurrency=concurrency,
                      label=f'repeat_audit:{directory.name}:c{concurrency}:{i}')
                      for concurrency in args.concurrency for i in range(args.repeats)]
        audits = [compare_executions(executions[0], r) for r in executions[1:]]
        errors = sum(bool(r.get('error')) for execution in executions for r in execution)
        stable = not errors and all(a['stable'] for a in audits)
        report['users'].append(dict(user_id=rows[0]['user_id'], audits=audits, errors=errors,
                                    code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                    stable=stable, executions=executions))
        report['stable'] = report['stable'] and stable
        atomic_json(args.output, report)
    print(f"{model}: stable={report['stable']}; audit={args.output}")
    if not report['stable']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
