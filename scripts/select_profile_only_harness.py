"""Select a per-user harness using only held-out historical profile tasks.

Evolution itself is run on ``profile_adaptation.jsonl``.  This optional final
selection compares the seed and evolved incumbent on
``profile_selection.jsonl``.  It never receives the official test row or its
target.  The selected source is written as ``selected_harness.py`` beside each
user's run artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="Profile-only historical selection tasks; never the official test file.",
    )
    parser.add_argument("--qa-url", default="http://gpu01:8000/v1")
    parser.add_argument("--qa-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-url", default="http://gpu01:18013/v1")
    parser.add_argument("--qa-concurrency", type=int, default=16)
    parser.add_argument("--user-workers", type=int, default=4)
    parser.add_argument("--max-users", type=int)
    args = parser.parse_args()

    rows = evo.load_jsonl(args.selection)
    groups = evo.grouped(rows)
    users = sorted(groups)
    if args.max_users:
        users = users[: args.max_users]
    if not users:
        raise ValueError("selection file has no users")
    if any(str(row.get("source_split", "")).startswith("test") is False for row in rows):
        raise ValueError("selection rows must be profile-only test-user history rows")

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
    if len(qa.embedding_client.embed(["profile-only selection preflight"])[0]) != 1024:
        raise RuntimeError("Embedding dimension differs from runtime contract")

    def one(user: str) -> dict:
        directory = args.run_dir / "users" / user_key(user)
        candidates = []
        for name in ("seed.py", "current_harness.py"):
            path = directory / name
            if path.exists():
                candidates.append((name.removesuffix(".py"), path))
        if not candidates:
            return {"user_id": user, "status": "missing_run_artifacts"}
        scores = {}
        predictions_by_name = {}
        for name, path in candidates:
            predictions = evo.evaluate_code(
                path.read_text(),
                groups[user],
                qa,
                qa_max_calls=8,
                qa_concurrency=min(8, len(groups[user])),
                label=f"{user_key(user)}:profile_selection:{name}",
            )
            summary = evo.summarize_rows(predictions, trace_limit=len(predictions))
            scores[name] = summary
            predictions_by_name[name] = predictions
            evo.write_jsonl(directory / f"{name}_selection_predictions.jsonl", predictions)

        # Ties deliberately keep seed: a more complex harness must earn its
        # selection on historical profile tasks.
        selected_name = max(
            scores,
            key=lambda name: (
                evo.weighted_score(scores[name]),
                name == "seed",
            ),
        )
        selected_path = directory / "selected_harness.py"
        shutil.copy2(directory / f"{selected_name}.py", selected_path)
        selection = {
            "user_id": user,
            "status": "selected",
            "selected": selected_name,
            "selection_rows": len(groups[user]),
            "official_test_target_used": False,
            "scores": scores,
            "selected_path": str(selected_path),
        }
        atomic_json(directory / "selection.json", selection)
        return selection

    results = {}
    with ThreadPoolExecutor(max_workers=max(1, args.user_workers)) as pool:
        futures = {pool.submit(one, user): user for user in users}
        for future in as_completed(futures):
            user = futures[future]
            try:
                result = future.result()
            except Exception as error:  # keep other users resumable
                result = {"user_id": user, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            results[user] = result
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "protocol": "strict_user_test_time_profile_only_selection",
        "run_dir": str(args.run_dir),
        "selection_file": str(args.selection),
        "users": len(users),
        "selected": sum(item.get("status") == "selected" for item in results.values()),
        "failed": sum(item.get("status") == "failed" for item in results.values()),
        "missing": sum(item.get("status") == "missing_run_artifacts" for item in results.values()),
        "official_test_target_used": False,
    }
    atomic_json(args.run_dir / "profile_only_selection_summary.json", {**summary, "users_detail": results})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["failed"] or summary["missing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
