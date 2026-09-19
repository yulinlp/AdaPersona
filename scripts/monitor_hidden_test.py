"""Side-channel per-round test monitor for profile-only harness evolution.

This process is deliberately outside evolve_per_user.py. It watches
operation-complete events, runs the current per-user harness on the official
test row, and writes aggregate scores to a sibling directory. Test targets
are held only by this scorer; no test row, prediction, score, or hypothesis is
written into an evolution run directory or passed to the Code Agent.
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
    from .evolve_per_user import LimitedQA, user_key
except ImportError:  # pragma: no cover
    import code_evolution as evo
    from embedding_client import EmbeddingClient
    from evolve_per_user import LimitedQA, user_key


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl_tolerant(path: Path) -> list[dict]:
    """Read complete lines while a writer may still be appending a record."""
    rows = []
    try:
        with path.open() as handle:
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
        raise ValueError(f"official test is empty: {path}")
    if any(len(rows) != 1 for rows in groups.values()):
        raise ValueError("hidden official test must contain exactly one row per user")
    return groups


def run_config(run_dir: Path, args) -> dict:
    config = read_json(run_dir / "run_config.json", {})
    return {
        "qa_url": config.get("qa_url", args.default_qa_url),
        "qa_model": config.get("qa_model", args.default_qa_model),
        "embedding_url": config.get("embedding_url", args.default_embedding_url),
        "label": run_dir.name,
    }


def discover_events(run_dir: Path, test_groups: dict[str, list[dict]]) -> list[dict]:
    """Reconstruct the incumbent source at every operation boundary."""
    users_dir = run_dir / "users"
    discovered = []
    if not users_dir.exists():
        return discovered
    for user_dir in sorted(p for p in users_dir.iterdir() if p.is_dir()):
        history = read_jsonl_tolerant(user_dir / "history.jsonl")
        state = read_json(user_dir / "state.json", {})
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
                "run": run_dir.name,
                "user_id": user_id,
                "iteration": 0,
                "operation": "seed",
                "accepted_candidate": "seed",
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
                "run": run_dir.name,
                "user_id": user_id,
                "iteration": int(event.get("iteration", 0)),
                "operation": str(event.get("operation", "")),
                "accepted_candidate": accepted,
                "code_path": str(current),
            })
    return discovered


def event_key(item: dict) -> str:
    return "|".join([
        item["run"], item["user_id"], str(item["iteration"]), item["operation"],
    ])


def make_clients(configs: dict[str, dict], args) -> dict[str, LimitedQA]:
    clients = {}
    for key, config in configs.items():
        with urllib.request.urlopen(
            config["qa_url"].rstrip("/") + "/models", timeout=15
        ) as response:
            models = json.load(response).get("data", [])
        if not any(item.get("id") == config["qa_model"] for item in models):
            raise RuntimeError(
                f"required hidden-test QA model {config['qa_model']} is not served "
                f"at {config['qa_url']}"
            )
        client = LimitedQA(
            config["qa_url"],
            config["qa_model"],
            concurrency=args.qa_concurrency,
            request_limit=args.qa_concurrency,
            timeout=args.qa_timeout,
            retries=1,
            chat_template_kwargs=(
                {"enable_thinking": False}
                if "qwen3" in config["qa_model"].lower()
                else None
            ),
        )
        client.embedding_client = EmbeddingClient(
            config["embedding_url"], truncate_prompt_tokens=2048
        )
        if len(client.embedding_client.embed(["hidden test preflight"])[0]) != 1024:
            raise RuntimeError("embedding dimension differs from runtime contract")
        clients[key] = client
    return clients


def evaluate_one(item: dict, test_groups: dict[str, list[dict]],
                 clients: dict[str, LimitedQA], args,
                 seed_scores: dict[tuple[str, str], float]) -> dict:
    row = test_groups[item["user_id"]][0]
    client = clients[item["run"]]
    predictions = evo.evaluate_code(
        Path(item["code_path"]).read_text(),
        [row],
        client,
        qa_max_calls=8,
        qa_concurrency=1,
        sample_timeout=args.sample_timeout,
        label=f"hidden:{item['run']}:{user_key(item['user_id'])}:"
              f"{item['iteration']}:{item['operation']}",
    )
    summary = evo.summarize_rows(predictions, trace_limit=0)
    score = float(summary["weighted_score_100"])
    seed = seed_scores.get((item["run"], item["user_id"]))
    return {
        "run": item["run"],
        "user_id": item["user_id"],
        "iteration": item["iteration"],
        "operation": item["operation"],
        "accepted_candidate": item.get("accepted_candidate"),
        "weighted_score_100": score,
        "gain_vs_seed_points": None if seed is None else round(score - seed, 3),
        "rouge1_100": summary["mean_rouge1_100"],
        "rougeL_100": summary["mean_rougeL_100"],
        "bleu_100": summary["mean_bleu_100"],
        "meteor_100": summary["mean_meteor_100"],
        "errors": summary["errors"],
        "qa_calls": summary["mean_qa_calls"],
        "code_path": item["code_path"],
        "target_used_only_by_shadow_scorer": True,
        "evolver_visibility": "none",
        "time": time.time(),
    }


def aggregate(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["run"], record["iteration"], record["operation"])].append(record)
    result = {}
    for (run, iteration, operation), values in sorted(grouped.items()):
        def mean(field):
            return round(statistics.mean(float(v[field]) for v in values), 3)

        gains = [
            float(v["gain_vs_seed_points"]) for v in values
            if v["gain_vs_seed_points"] is not None
        ]
        result[f"{run}|{iteration}|{operation}"] = {
            "run": run,
            "iteration": iteration,
            "operation": operation,
            "users_scored": len(values),
            "weighted_score_100": mean("weighted_score_100"),
            "gain_vs_seed_points": round(statistics.mean(gains), 3) if gains else None,
            "rouge1_100": mean("rouge1_100"),
            "rougeL_100": mean("rougeL_100"),
            "bleu_100": mean("bleu_100"),
            "meteor_100": mean("meteor_100"),
            "errors": sum(int(v["errors"]) for v in values),
            "time": max(v["time"] for v in values),
        }
    return result


def score_batch(pending, test_groups, clients, args, seed_scores, records,
                seen, scores_path):
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=max(1, args.user_workers)) as pool:
        futures = [
            pool.submit(evaluate_one, item, test_groups, clients, args, seed_scores)
            for item in pending
        ]
        for future in as_completed(futures):
            record = future.result()
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with scores_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            seen.add(event_key(record))
            if record["operation"] == "seed":
                seed_scores[(record["run"], record["user_id"])] = record["weighted_score_100"]
            print(json.dumps({
                "event": "hidden_test_score",
                "run": record["run"],
                "user_id": record["user_id"],
                "iteration": record["iteration"],
                "operation": record["operation"],
                "weighted_100": record["weighted_score_100"],
                "gain_vs_seed": record["gain_vs_seed_points"],
            }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--user-workers", type=int, default=4)
    parser.add_argument("--qa-concurrency", type=int, default=8)
    parser.add_argument("--qa-timeout", type=float, default=120)
    parser.add_argument("--sample-timeout", type=float, default=300)
    parser.add_argument("--default-qa-url", default="http://gpu01:8000/v1")
    parser.add_argument("--default-qa-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--default-embedding-url", default="http://gpu01:18013/v1")
    args = parser.parse_args()
    test_groups = load_test_groups(args.test)
    configs = {run.name: run_config(run, args) for run in args.run_dir}
    clients = make_clients(configs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "per_round_scores.jsonl"
    records = read_jsonl_tolerant(scores_path)
    seen = {event_key(record) for record in records}
    seed_scores = {
        (record["run"], record["user_id"]): float(record["weighted_score_100"])
        for record in records if record["operation"] == "seed"
    }
    atomic_json(args.output_dir / "protocol.json", {
        "official_test": str(args.test),
        "target_used_only_by_shadow_scorer": True,
        "evolver_visibility": "none",
        "selection_uses_hidden_test": False,
        "runs": [str(path) for path in args.run_dir],
    })
    print(json.dumps({
        "event": "hidden_test_monitor_started",
        "runs": [str(path) for path in args.run_dir],
        "users": len(test_groups),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False), flush=True)

    while True:
        pending = []
        for run in args.run_dir:
            for item in discover_events(run, test_groups):
                if event_key(item) not in seen:
                    pending.append(item)
        score_batch(
            [item for item in pending if item["operation"] == "seed"],
            test_groups, clients, args, seed_scores, records, seen, scores_path,
        )
        score_batch(
            [item for item in pending if item["operation"] != "seed"],
            test_groups, clients, args, seed_scores, records, seen, scores_path,
        )
        if pending:
            atomic_json(args.output_dir / "aggregate.json", aggregate(records))
            atomic_json(args.output_dir / "latest.json", records[-1] if records else {})
        if args.once:
            break
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
