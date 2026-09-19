"""Side-channel per-round official-test monitor for LaMP-2/3/4/5.

This process is deliberately outside ``evolve_per_user.py``.  It watches
operation-complete checkpoints, evaluates the frozen current harness on the
official current task, and writes only to a sibling output directory.  The
official target is held by this scorer and is never sent to the harness or
the Code Agent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import time
import urllib.request

try:
    from . import code_evolution as evo
    from .embedding_client import EmbeddingClient
    from .evolve_per_user import LimitedQA, user_key, LongLaMPAdapter
    from .lamp_tasks import TASKS, LampTaskAdapter
except ImportError:  # pragma: no cover
    import code_evolution as evo
    from embedding_client import EmbeddingClient
    from evolve_per_user import LimitedQA, user_key, LongLaMPAdapter
    from lamp_tasks import TASKS, LampTaskAdapter


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl_tolerant(path: Path) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except FileNotFoundError:
        pass
    return rows


def load_test_groups(path: Path) -> dict[str, list[dict]]:
    groups = evo.grouped(evo.load_jsonl(path))
    if not groups:
        raise ValueError(f"official LaMP test is empty: {path}")
    if any(len(rows) != 1 for rows in groups.values()):
        raise ValueError("official LaMP test must contain exactly one row per user")
    return groups


def discover_events(run_dir: Path, test_groups: dict[str, list[dict]]) -> list[dict]:
    users_dir = run_dir / "users"
    discovered = []
    if not users_dir.exists():
        return discovered
    for user_dir in sorted(path for path in users_dir.iterdir() if path.is_dir()):
        history = read_jsonl_tolerant(user_dir / "history.jsonl")
        state = read_json(user_dir / "state.json", {})
        committed = state.get('last_operation_event')
        if committed and not any(e.get('event') == 'operation_complete' and
                                e.get('iteration') == committed['iteration'] for e in history):
            history.append(committed)
        user_id = str(state.get("user_id", ""))
        if not user_id:
            for event in history:
                if event.get("user_id"):
                    user_id = str(event["user_id"])
                    break
        if not user_id or user_id not in test_groups:
            continue
        current = user_dir / "seed.py"
        if current.exists():
            discovered.append({
                "run": run_dir.name, "user_id": user_id, "iteration": 0,
                "operation": "seed", "accepted_candidate": "seed",
                "code_path": str(current),
            })
        for event in history:
            if event.get("event") != "operation_complete":
                continue
            accepted = event.get("accepted_candidate")
            if accepted:
                candidate = user_dir / f"{accepted}.py"
                if not candidate.exists():
                    continue
                current = candidate
            if not current.exists():
                continue
            discovered.append({
                "run": run_dir.name, "user_id": user_id,
                "iteration": int(event.get("iteration", 0)),
                "operation": str(event.get("operation", "")),
                "accepted_candidate": accepted, "code_path": str(current),
            })
    return discovered


def event_key(item: dict) -> str:
    return "|".join([
        item["run"], item["user_id"], str(item["iteration"]), item["operation"],
    ])


def make_clients(run_dirs: list[Path], args) -> dict[str, LimitedQA]:
    clients = {}
    for run_dir in run_dirs:
        config = read_json(run_dir / "run_config.json", {})
        if not config:
            raise RuntimeError('Run configuration is not ready')
        if config.get('benchmark', 'longlamp') != args.benchmark or (
                args.benchmark == 'lamp' and config.get('task') != args.task):
            raise ValueError('Monitor benchmark/task does not match run configuration')
        qa_url = config.get("qa_url", args.qa_url)
        qa_model = config.get("qa_model", args.qa_model)
        embedding_url = config.get("embedding_url", args.embedding_url)
        with urllib.request.urlopen(qa_url.rstrip("/") + "/models", timeout=15) as response:
            models = json.load(response).get("data", [])
        if not any(item.get("id") == qa_model for item in models):
            raise RuntimeError(f"required QA model {qa_model} is not served at {qa_url}")
        client = LimitedQA(
            qa_url, qa_model, concurrency=args.qa_concurrency,
            request_limit=args.qa_concurrency, timeout=args.qa_timeout, retries=1,
            chat_template_kwargs={"enable_thinking": False}
            if "qwen3" in qa_model.lower() else None,
        )
        client.embedding_client = EmbeddingClient(embedding_url, truncate_prompt_tokens=2048)
        if len(client.embedding_client.embed(["LaMP hidden test preflight"])[0]) != 1024:
            raise RuntimeError("embedding dimension differs from runtime contract")
        clients[run_dir.name] = client
    return clients


def evaluate_one(item: dict, groups: dict[str, list[dict]], clients, adapter, args,
                 seed_scores: dict[tuple[str, str], float]) -> dict:
    row = groups[item["user_id"]][0]
    predictions = evo.evaluate_code(
        Path(item["code_path"]).read_text(encoding="utf-8"), [row], clients[item["run"]],
        qa_max_calls=8, qa_concurrency=1, sample_timeout=args.sample_timeout,
        label=f"lamp_hidden:{item['run']}:{user_key(item['user_id'])}:"
              f"{item['iteration']}:{item['operation']}",
    )
    summary = adapter.summarize_rows(predictions, trace_limit=0)
    seed = seed_scores.get((item["run"], item["user_id"]))
    record = {
        "run": item["run"], "user_id": item["user_id"],
        "iteration": item["iteration"], "operation": item["operation"],
        "accepted_candidate": item.get("accepted_candidate"),
        "weighted_score_100": summary["weighted_score_100"],
        "gain_vs_seed_points": None if seed is None else round(
            summary["weighted_score_100"] - seed, 3),
        "errors": summary["errors"], "qa_calls": summary.get("mean_qa_calls", 0.0),
        "code_path": item["code_path"], "target_used_only_by_shadow_scorer": True,
        "evolver_visibility": "none", "time": time.time(),
    }
    record['prediction'] = predictions[0].get('prediction', '')
    record['error'] = predictions[0].get('error', '')
    record['code_sha256'] = __import__('hashlib').sha256(Path(item['code_path']).read_bytes()).hexdigest()
    for diagnostic in ('mae', 'rmse', 'macro_f1', 'invalid_predictions'):
        if diagnostic in summary:
            record[diagnostic] = summary[diagnostic]
    for metric in adapter.metric_names:
        record[f"{metric}_100"] = summary.get(f"mean_{metric}_100", 0.0)
    return record


def aggregate(records: list[dict], metric_names: tuple[str, ...]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["run"], record["iteration"], record["operation"])].append(record)
    result = {}
    for (run, iteration, operation), values in sorted(grouped.items()):
        def mean(field):
            return round(statistics.mean(float(value[field]) for value in values), 3)
        gains = [float(value["gain_vs_seed_points"]) for value in values
                 if value["gain_vs_seed_points"] is not None]
        result[f"{run}|{iteration}|{operation}"] = {
            "run": run, "iteration": iteration, "operation": operation,
            "users_scored": len(values), "weighted_score_100": mean("weighted_score_100"),
            "gain_vs_seed_points": round(statistics.mean(gains), 3) if gains else None,
            "errors": sum(int(value["errors"]) for value in values),
            **{f"{metric}_100": mean(f"{metric}_100") for metric in metric_names},
            "time": max(value["time"] for value in values),
        }
    return result


def score_batch(pending, groups, clients, adapter, args, seed_scores, records, seen, scores_path):
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=max(1, args.user_workers)) as pool:
        futures = [pool.submit(evaluate_one, item, groups, clients, adapter, args, seed_scores)
                   for item in pending]
        for future in as_completed(futures):
            record = future.result()
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with scores_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            seen.add(event_key(record))
            if record["operation"] == "seed":
                seed_scores[(record["run"], record["user_id"])] = record["weighted_score_100"]
            print(json.dumps({
                "event": "lamp_hidden_test_score", "run": record["run"],
                "user_id": record["user_id"], "iteration": record["iteration"],
                "operation": record["operation"],
                "weighted_100": record["weighted_score_100"],
                "gain_vs_seed": record["gain_vs_seed_points"],
            }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument('--benchmark', choices=('lamp', 'longlamp'), default='lamp')
    parser.add_argument("--task", type=int, choices=TASKS)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--user-workers", type=int, default=4)
    parser.add_argument("--qa-concurrency", type=int, default=8)
    parser.add_argument("--qa-timeout", type=float, default=120)
    parser.add_argument("--sample-timeout", type=float, default=300)
    parser.add_argument("--qa-url", default="http://gpu01:8000/v1")
    parser.add_argument("--qa-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-url", default="http://gpu01:18013/v1")
    args = parser.parse_args()
    if args.benchmark == 'lamp' and args.task is None:
        parser.error('--task required for LaMP')
    if any(args.output_dir.resolve() == run.resolve() or run.resolve() in args.output_dir.resolve().parents
           for run in args.run_dir):
        parser.error('Test monitoring output must be outside the evolution directory')
    adapter = LampTaskAdapter(args.task) if args.benchmark == 'lamp' else LongLaMPAdapter()
    metric_config = adapter.preflight()
    groups = load_test_groups(args.test)
    clients = make_clients(args.run_dir, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "per_round_scores.jsonl"
    records = [r for r in read_jsonl_tolerant(scores_path) if not r.get('errors')]
    seen = {event_key(record) for record in records}
    seed_scores = {
        (record["run"], record["user_id"]): float(record["weighted_score_100"])
        for record in records if record["operation"] == "seed"
    }
    atomic_json(args.output_dir / "protocol.json", {
        "benchmark": "LaMP", "task": args.task, "official_test": str(args.test),
        "target_used_only_by_shadow_scorer": True, "evolver_visibility": "none",
        "selection_uses_hidden_test": False, "metric_config": metric_config,
        "runs": [str(path) for path in args.run_dir],
    })
    print(json.dumps({
        "event": "lamp_hidden_test_monitor_started", "task": args.task,
        "runs": [str(path) for path in args.run_dir], "users": len(groups),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False), flush=True)

    while True:
        pending = []
        for run in args.run_dir:
            pending.extend(item for item in discover_events(run, groups)
                           if event_key(item) not in seen)
        score_batch([item for item in pending if item["operation"] == "seed"],
                    groups, clients, adapter, args, seed_scores, records, seen, scores_path)
        score_batch([item for item in pending if item["operation"] != "seed"],
                    groups, clients, adapter, args, seed_scores, records, seen, scores_path)
        if pending:
            atomic_json(args.output_dir / "aggregate.json", aggregate(records, adapter.metric_names))
            atomic_json(args.output_dir / "latest.json", records[-1] if records else {})
        if args.once:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
