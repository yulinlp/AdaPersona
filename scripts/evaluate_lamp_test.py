"""Score evolved LaMP-2/3/4/5 harnesses on the official current task.

The test target is loaded by this outer scorer only.  ``evaluate_code`` strips
the target before invoking ``run(row, qa)``, so neither the harness nor the
frozen answer model can observe the official answer.  This script does not
write anything into an evolution run directory.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

try:
    from . import code_evolution as evo
    from .embedding_client import EmbeddingClient
    from .evolve_per_user import LimitedQA, user_key
    from .lamp_tasks import TASKS, LampTaskAdapter
except ImportError:  # pragma: no cover
    import code_evolution as evo
    from embedding_client import EmbeddingClient
    from evolve_per_user import LimitedQA, user_key
    from lamp_tasks import TASKS, LampTaskAdapter


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def infer_task(run_dir: Path, requested: int | None) -> int:
    config = run_dir / "run_config.json"
    if config.exists():
        value = json.loads(config.read_text(encoding="utf-8")).get("task")
        if str(value).isdigit() and int(value) in TASKS:
            if requested is not None and int(requested) != int(value):
                raise ValueError('Requested task differs from run configuration')
            return int(value)
    if requested is not None:
        return int(requested)
    raise ValueError("--task is required when it cannot be inferred from run_config.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--task", type=int, choices=TASKS)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=("current", "seed", "all"), default="all")
    parser.add_argument("--qa-url")
    parser.add_argument("--qa-model")
    parser.add_argument("--embedding-url", default="http://gpu01:18013/v1")
    parser.add_argument("--qa-concurrency", type=int, default=32)
    parser.add_argument("--user-workers", type=int, default=8)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()
    configs = [json.loads((run/'run_config.json').read_text()) for run in args.run_dir]
    models = {(c['qa_url'], c['qa_model']) for c in configs}
    if len(models) != 1:
        parser.error('Evaluate different answer-model runs separately')
    url, model = models.pop()
    if (args.qa_url and args.qa_url != url) or (args.qa_model and args.qa_model != model):
        parser.error('Answer model must match the evolution run')
    args.qa_url, args.qa_model = url, model
    if any(run.resolve() == args.output_dir.resolve() or run.resolve() in args.output_dir.resolve().parents
           for run in args.run_dir):
        parser.error('Test outputs must be outside the evolution directories')

    tasks = {infer_task(run_dir, args.task) for run_dir in args.run_dir}
    if len(tasks) != 1:
        raise ValueError("all run directories must belong to the same LaMP task")
    task = tasks.pop()
    test_path = args.test or Path(f"data/experiments/lamp_user_rsi/lamp_{task}/test.jsonl")
    test_rows = evo.load_jsonl(test_path)
    groups = evo.grouped(test_rows)
    if any(len(items) != 1 for items in groups.values()):
        raise ValueError("official LaMP user test must contain exactly one row per user")
    if not test_rows:
        raise ValueError("official LaMP test is empty")
    users = sorted(groups)
    if args.max_users:
        users = users[: max(0, int(args.max_users))]
    groups = {user: groups[user] for user in users}
    adapter = LampTaskAdapter(task)
    metric_config = adapter.preflight()

    with urlopen(args.qa_url.rstrip("/") + "/models", timeout=15) as response:
        models = json.load(response)["data"]
    qa_model = next((item for item in models if item.get("id") == args.qa_model), None)
    if qa_model is None:
        raise RuntimeError(f"Required frozen QA model {args.qa_model} is not served")
    qa = LimitedQA(
        args.qa_url,
        args.qa_model,
        concurrency=args.qa_concurrency,
        request_limit=args.qa_concurrency,
        timeout=120,
        retries=1,
        chat_template_kwargs={"enable_thinking": False}
        if "qwen3" in args.qa_model.lower() else None,
    )
    qa.embedding_client = EmbeddingClient(args.embedding_url, truncate_prompt_tokens=2048)
    if len(qa.embedding_client.embed(["LaMP official test preflight"])[0]) != 1024:
        raise RuntimeError("Embedding dimension differs from runtime contract")

    run_results = {}
    for run_dir in args.run_dir:
        run_label = run_dir.name
        candidates = ["current", "seed"] if args.candidate == "all" else [args.candidate]
        for candidate in candidates:
            prediction_rows = []

            def one(item):
                user, rows = item
                row = rows[0]
                directory = run_dir / "users" / user_key(user)
                path = directory / "seed.py" if candidate == "seed" else directory / "current_harness.py"
                if not path.exists():
                    return [{
                        "user_id": user,
                        "sample_id": row.get("sample_id", ""),
                        "input": row.get("input", ""),
                        "target": row.get("target", ""),
                        "prediction": "",
                        "profile": row.get("profile", []),
                        "source_split": row.get("source_split", "official_test"),
                        "error": f"missing harness: {path}",
                    }]
                return evo.evaluate_code(
                    path.read_text(encoding="utf-8"),
                    [row],
                    qa,
                    qa_max_calls=8,
                    qa_concurrency=1,
                    label=f"{run_label}:{candidate}:{user_key(user)}:official_test",
                )

            with ThreadPoolExecutor(max_workers=max(1, args.user_workers)) as pool:
                futures = [pool.submit(one, item) for item in groups.items()]
                for future in as_completed(futures):
                    prediction_rows.extend(future.result())
            prediction_rows.sort(key=lambda row: str(row.get("user_id", "")))
            summary = adapter.summarize_rows(prediction_rows, trace_limit=20)
            summary.update({
                "run_dir": str(run_dir),
                "run_label": run_label,
                "candidate": candidate,
                "task": task,
                "official_test_rows": len(prediction_rows),
                "official_test_target_used_during_runtime": False,
                "adaptation_protocol": "strict_user_test_profile_only_historical_target_adaptation",
            })
            key = f"{run_label}:{candidate}"
            run_results[key] = summary
            output_base = args.output_dir / run_label
            evo.write_jsonl(output_base / f"{candidate}_predictions.jsonl", prediction_rows)
            atomic_json(output_base / f"{candidate}_summary.json", summary)
            print(json.dumps({
                "event": "lamp_official_test_complete",
                "task": task,
                "run": run_label,
                "candidate": candidate,
                "rows": len(prediction_rows),
                "weighted_100": summary["weighted_score_100"],
                **{f"{metric}_100": summary.get(f"mean_{metric}_100", 0.0)
                   for metric in adapter.metric_names},
                "errors": summary["errors"],
            }, ensure_ascii=False), flush=True)

    report = {
        "benchmark": "LaMP",
        "task": task,
        "protocol": "strict_user_test_profile_only_official_test",
        "test_file": str(test_path),
        "users": len(groups),
        "runs": [str(path) for path in args.run_dir],
        "target_available_only_to_outer_scorer": True,
        "target_sent_to_runtime": False,
        "metric_config": metric_config,
        "results": run_results,
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
