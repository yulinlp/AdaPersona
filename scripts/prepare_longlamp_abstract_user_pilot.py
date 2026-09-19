"""Build a strict user-split LongLaMP abstract-generation pilot.

The official ``abstract_generation_user`` configuration has disjoint users in
train, validation, and test.  Training rows may be used to evolve the shared
module library and meta-policy.  Validation/test rows are kept as unseen
users: their profile and current task are available for routing, but their
target abstract is never fed back into adaptation or selection.

For training users we optionally add leave-one-out profile tasks.  These make
the training side large enough for RSI while preserving the same
profile-only information boundary used at inference time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_SOURCE = Path("data/benchmarks/LongLaMP/abstract_generation_user")

STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on",
    "or", "the", "to", "via", "with", "using", "based", "new", "study",
}


def stable_rank(seed: int, user: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{user}".encode()).digest()[:8], "big")


def read_users(source: Path, split: str) -> set[str]:
    users: set[str] = set()
    for path in sorted(source.glob(f"{split}-*.parquet")):
        table = pq.ParquetFile(path).read(columns=["name"])
        users.update(str(value) for value in table["name"].to_pylist() if value is not None)
    return users


def read_selected(source: Path, split: str, selected: set[str]) -> dict[str, dict[str, Any]]:
    """Read only selected users without materializing the whole split at once."""
    rows: dict[str, dict[str, Any]] = {}
    columns = ["name", "input", "output", "profile"]
    for path in sorted(source.glob(f"{split}-*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            for row in batch.to_pylist():
                user = str(row.get("name", ""))
                if user in selected:
                    rows[user] = {
                        "user_id": user,
                        "input": str(row.get("input", "")),
                        "target": str(row.get("output", "")),
                        "profile": row.get("profile") or [],
                        "source_split": split,
                    }
    return rows


def abstract_input(title: str) -> str:
    """Create a title-only leave-one-out task from one user's history."""
    words: list[str] = []
    seen: set[str] = set()
    for word in title.replace("-", " ").split():
        clean = "".join(char for char in word if char.isalnum())
        lowered = clean.lower()
        if len(clean) >= 4 and lowered not in STOPWORDS and lowered not in seen:
            words.append(clean)
            seen.add(lowered)
        if len(words) >= 5:
            break
    if not words:
        return f'Generate an abstract for the title "{title}".'
    items = "\n".join(f"{index}. {word}" for index, word in enumerate(words, 1))
    return f'Generate an abstract for the title "{title}" using the following items: {items}'


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _profile_tasks(
    user: str,
    profile: list[dict[str, Any]],
    *,
    seed: int,
    adaptation_count: int,
    holdout_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    if adaptation_count < 1:
        raise ValueError("adaptation-per-user must be positive")
    if holdout_count < 0:
        raise ValueError("holdout-per-user cannot be negative")
    ordered = sorted(
        profile,
        key=lambda item: (int(item.get("year", 0)), str(item.get("id", ""))),
    )
    # Keep a prefix as historical evidence and sample later records as
    # synthetic current tasks.  The held-out target is removed from evidence.
    candidate_start = min(8, max(0, len(ordered) - 1))
    indices = list(range(candidate_start, len(ordered)))
    rng = random.Random(stable_rank(seed, user))
    rng.shuffle(indices)
    needed = max(0, adaptation_count - 1) + holdout_count
    if len(indices) < needed:
        raise ValueError(
            f"user {user!r} has only {len(indices)} profile candidates, needs {needed}"
        )
    adaptation_indices = indices[: max(0, adaptation_count - 1)]
    holdout_indices = indices[max(0, adaptation_count - 1) : max(0, adaptation_count - 1) + holdout_count]

    adaptation: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    adaptation_ids: list[str] = []
    holdout_ids: list[str] = []
    for idx in adaptation_indices:
        item = ordered[idx]
        item_id = str(item.get("id", idx))
        row_id = f"profile:{user}:{item_id}"
        adaptation.append({
            "user_id": user,
            "sample_id": row_id,
            "input": abstract_input(str(item.get("title", ""))),
            "target": str(item.get("abstract", "")),
            "profile": ordered[:idx],
            "source_split": "train_profile",
            "origin": "leave_one_out_profile",
        })
        adaptation_ids.append(row_id)
    for idx in holdout_indices:
        item = ordered[idx]
        item_id = str(item.get("id", idx))
        row_id = f"holdout:{user}:{item_id}"
        holdout.append({
            "user_id": user,
            "sample_id": row_id,
            "input": abstract_input(str(item.get("title", ""))),
            "target": str(item.get("abstract", "")),
            "profile": ordered[:idx],
            "source_split": "train_profile_holdout",
            "origin": "leave_one_out_profile",
        })
        holdout_ids.append(row_id)
    return adaptation, holdout, adaptation_ids, holdout_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare disjoint-user LongLaMP abstract pilot files.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/longlamp_abstract_user_rsi"))
    parser.add_argument("--max-train-users", type=int, default=256)
    parser.add_argument("--max-val-users", type=int, default=256)
    parser.add_argument("--max-test-users", type=int, default=256)
    parser.add_argument("--adaptation-per-user", type=int, default=8)
    parser.add_argument("--holdout-per-user", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    if min(args.max_train_users, args.max_val_users, args.max_test_users) < 1:
        raise ValueError("all max-* users values must be positive")
    if args.adaptation_per_user < 1:
        raise ValueError("adaptation-per-user must be positive")
    if args.holdout_per_user < 0:
        raise ValueError("holdout-per-user cannot be negative")

    all_users = {split: read_users(args.source_dir, split) for split in ("train", "val", "test")}
    overlaps = {
        f"{left}-{right}": sorted(all_users[left] & all_users[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    bad_overlap = {key: value for key, value in overlaps.items() if value}
    if bad_overlap:
        raise ValueError(f"source is not a strict user split: { {key: len(value) for key, value in bad_overlap.items()} }")

    limits = {
        "train": args.max_train_users,
        "val": args.max_val_users,
        "test": args.max_test_users,
    }
    selected = {
        split: set(sorted(all_users[split], key=lambda user: stable_rank(args.seed, user))[: limits[split]])
        for split in ("train", "val", "test")
    }
    source_rows = {
        split: read_selected(args.source_dir, split, selected[split])
        for split in ("train", "val", "test")
    }

    adaptation: list[dict[str, Any]] = []
    train_holdout: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for user in sorted(selected["train"], key=lambda value: stable_rank(args.seed, value)):
        row = source_rows["train"].get(user)
        if row is None:
            continue
        adaptation.append({
            **row,
            "sample_id": f"train:{user}",
            "source_split": "train_user",
            "origin": "user_train",
        })
        try:
            profile_adaptation, profile_holdout, adaptation_ids, holdout_ids = _profile_tasks(
                user,
                list(row.get("profile") or []),
                seed=args.seed,
                adaptation_count=args.adaptation_per_user,
                holdout_count=args.holdout_per_user,
            )
        except ValueError:
            profile_adaptation, profile_holdout, adaptation_ids, holdout_ids = [], [], [], []
        adaptation.extend(profile_adaptation)
        train_holdout.extend(profile_holdout)
        manifest.append({
            "user_id": user,
            "split": "train",
            "n_profile": len(row.get("profile") or []),
            "adaptation_ids": [f"train:{user}"] + adaptation_ids,
            "holdout_ids": holdout_ids,
        })

    validation = []
    test_rows = []
    for split, destination, prefix in (("val", validation, "val"), ("test", test_rows, "test")):
        for user in sorted(selected[split], key=lambda value: stable_rank(args.seed, value)):
            row = source_rows[split].get(user)
            if row is None:
                continue
            destination.append({
                **row,
                "sample_id": f"{prefix}:{user}",
                "source_split": f"{split}_user",
                "origin": f"user_{split}",
            })
            manifest.append({
                "user_id": user,
                "split": split,
                "n_profile": len(row.get("profile") or []),
                "sample_id": f"{prefix}:{user}",
            })

    args.output.mkdir(parents=True, exist_ok=True)
    # Canonical names consumed by the V5 and baseline runners.
    write_jsonl(args.output / "evolve_train.jsonl", adaptation)
    write_jsonl(args.output / "evolve_val.jsonl", validation)
    write_jsonl(args.output / "inner_holdout.jsonl", train_holdout)
    write_jsonl(args.output / "test.jsonl", test_rows)
    # Descriptive aliases make the boundary obvious when inspecting a run.
    write_jsonl(args.output / "adaptation.jsonl", adaptation)
    write_jsonl(args.output / "dev.jsonl", validation)
    write_jsonl(args.output / "train_holdout.jsonl", train_holdout)
    write_jsonl(args.output / "users.jsonl", manifest)

    stats = {
        "source": str(args.source_dir),
        "split_mode": "strict_user",
        "seed": args.seed,
        "users": {split: len(selected[split]) for split in ("train", "val", "test")},
        "rows": {
            "evolve_train": len(adaptation),
            "evolve_val": len(validation),
            "inner_holdout": len(train_holdout),
            "test": len(test_rows),
        },
        "adaptation_per_user_requested": args.adaptation_per_user,
        "holdout_per_user": args.holdout_per_user,
        "profile_size": {
            "train_mean": sum(len(row.get("profile") or []) for row in source_rows["train"].values()) / max(1, len(source_rows["train"])),
            "val_mean": sum(len(row.get("profile") or []) for row in source_rows["val"].values()) / max(1, len(source_rows["val"])),
            "test_mean": sum(len(row.get("profile") or []) for row in source_rows["test"].values()) / max(1, len(source_rows["test"])),
        },
    }
    (args.output / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
