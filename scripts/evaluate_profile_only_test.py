"""Evaluate frozen per-user harnesses once on official strict-user test rows.

The target is retained by this outer scorer for metrics only.  Each harness is
executed through ``code_evolution.evaluate_code``, which passes ``runtime_row``
without the target, so test feedback cannot enter the harness or adaptation.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/experiments/longlamp_abstract_user_rsi_tta/test.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=("selected", "current", "seed", "all"), default="all")
    parser.add_argument("--qa-url", default="http://gpu01:8000/v1")
    parser.add_argument("--qa-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-url", default="http://gpu01:18013/v1")
    parser.add_argument("--qa-concurrency", type=int, default=32)
    parser.add_argument("--user-workers", type=int, default=8)
    args = parser.parse_args()

    test_rows = evo.load_jsonl(args.test)
    groups = evo.grouped(test_rows)
    if any(len(items) != 1 for items in groups.values()):
        raise ValueError("official strict-user test must contain exactly one row per user")
    if not test_rows:
        raise ValueError("official test is empty")

    with __import__("urllib.request", fromlist=["urlopen"]).urlopen(
        args.qa_url.rstrip("/") + "/models", timeout=15
    ) as response:
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
        chat_template_kwargs={"enable_thinking": False} if "qwen3" in args.qa_model.lower() else None,
    )
    qa.embedding_client = EmbeddingClient(args.embedding_url, truncate_prompt_tokens=2048)
    if len(qa.embedding_client.embed(["profile-only test preflight"])[0]) != 1024:
        raise RuntimeError("Embedding dimension differs from runtime contract")

    run_results = {}
    for run_dir in args.run_dir:
        run_label = run_dir.name
        if args.candidate == "all":
            candidates = ["selected", "current", "seed"]
        else:
            candidates = [args.candidate]
        for candidate in candidates:
            prediction_rows = []

            def one(item):
                user, rows = item
                row = rows[0]
                directory = run_dir / "users" / user_key(user)
                path = directory / f"{candidate}_harness.py" if candidate != "seed" else directory / "seed.py"
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
                    path.read_text(),
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
            summary = evo.summarize_rows(prediction_rows, trace_limit=20)
            summary.update({
                "run_dir": str(run_dir),
                "run_label": run_label,
                "candidate": candidate,
                "official_test_rows": len(prediction_rows),
                "official_test_target_used_during_runtime": False,
                "adaptation_protocol": "strict_user_test_time_profile_only_self_supervised",
            })
            run_results[f"{run_label}:{candidate}"] = summary
            output_base = args.output_dir / run_label
            evo.write_jsonl(output_base / f"{candidate}_predictions.jsonl", prediction_rows)
            atomic_json(output_base / f"{candidate}_summary.json", summary)
            print(json.dumps({
                "event": "official_test_complete",
                "run": run_label,
                "candidate": candidate,
                "rows": len(prediction_rows),
                "weighted_100": summary["weighted_score_100"],
                "rouge1_100": summary["mean_rouge1_100"],
                "rougeL_100": summary["mean_rougeL_100"],
                "bleu_100": summary["mean_bleu_100"],
                "meteor_100": summary["mean_meteor_100"],
                "errors": summary["errors"],
            }, ensure_ascii=False), flush=True)

    report = {
        "protocol": "strict_user_test_time_profile_only_official_test",
        "test_file": str(args.test),
        "users": len(groups),
        "runs": [str(path) for path in args.run_dir],
        "target_available_only_to_outer_scorer": True,
        "target_sent_to_runtime": False,
        "results": run_results,
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
