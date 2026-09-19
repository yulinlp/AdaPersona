"""Prepare profile-only self-supervised adaptation for strict user-split test users.

The official test task is never copied into the adaptation files.  For each
test user's historical profile, this script creates deterministic leave-one-out
abstract tasks whose targets are historical profile abstracts.  These targets
are pseudo-labels from user-provided history, not the current test target.

This is intentionally called *profile-only self-supervised* rather than fully
unsupervised: an optimizer still needs a score, so it may compare generated
outputs with held-out historical profile abstracts.  The real test target is
kept only in ``test.jsonl`` for the final offline scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def stable_rank(seed: int, user: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{user}".encode()).digest()[:8], "big")


def abstract_input(title: str) -> str:
    stopwords = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on",
        "or", "the", "to", "via", "with", "using", "based", "new", "study",
    }
    words: list[str] = []
    seen: set[str] = set()
    for word in title.replace("-", " ").split():
        clean = "".join(char for char in word if char.isalnum())
        lowered = clean.lower()
        if len(clean) >= 4 and lowered not in stopwords and lowered not in seen:
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


def profile_tasks(
    user: str,
    profile: list[dict[str, Any]],
    *,
    seed: int,
    adaptation_count: int,
    selection_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        profile,
        key=lambda item: (int(item.get("year", 0)), str(item.get("id", ""))),
    )
    # Keep an initial prefix as context.  Later profile documents become
    # pseudo-current tasks and are never included in their own profile.
    candidate_start = min(8, max(0, len(ordered) - 1))
    indices = list(range(candidate_start, len(ordered)))
    rng = random.Random(stable_rank(seed, user))
    rng.shuffle(indices)
    needed = adaptation_count + selection_count
    if len(indices) < needed:
        raise ValueError(f"{user!r}: profile has {len(indices)} candidates, needs {needed}")

    def make(indices_for_split: list[int], split: str) -> list[dict[str, Any]]:
        rows = []
        for index in indices_for_split:
            item = ordered[index]
            item_id = str(item.get("id", index))
            rows.append({
                "user_id": user,
                "sample_id": f"profile_{split}:{user}:{item_id}",
                "input": abstract_input(str(item.get("title", ""))),
                # This is a historical profile pseudo-label.  It is never the
                # official test target and is not sent to the runtime harness.
                "target": str(item.get("abstract", "")),
                "profile": ordered[:index],
                "source_split": f"test_profile_{split}",
                "origin": "profile_only_self_supervised_leave_one_out",
            })
        return rows

    adaptation_indices = indices[:adaptation_count]
    selection_indices = indices[adaptation_count:adaptation_count + selection_count]
    manifest = {
        "user_id": user,
        "profile_size": len(ordered),
        "adaptation_ids": [f"profile_adaptation:{user}:{ordered[i].get('id', i)}" for i in adaptation_indices],
        "selection_ids": [f"profile_selection:{user}:{ordered[i].get('id', i)}" for i in selection_indices],
        "official_test_id": f"test:{user}",
    }
    return make(adaptation_indices, "adaptation"), make(selection_indices, "selection"), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-test",
        type=Path,
        default=Path("data/experiments/longlamp_abstract_user_rsi/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/longlamp_abstract_user_rsi_tta"),
    )
    parser.add_argument("--adaptation-per-user", type=int, default=8)
    parser.add_argument("--selection-per-user", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.adaptation_per_user < 1 or args.selection_per_user < 0:
        raise ValueError("adaptation-per-user must be positive and selection-per-user nonnegative")

    test_rows = []
    with args.source_test.open() as handle:
        for line in handle:
            if line.strip():
                test_rows.append(json.loads(line))
    by_user: dict[str, dict[str, Any]] = {}
    for row in test_rows:
        user = str(row["user_id"])
        if user in by_user:
            raise ValueError(f"Expected one official test row per user, found duplicate: {user}")
        by_user[user] = row

    adaptation: list[dict[str, Any]] = []
    selection: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for user in sorted(by_user, key=lambda value: stable_rank(args.seed, value)):
        try:
            adapt_rows, select_rows, user_manifest = profile_tasks(
                user,
                list(by_user[user].get("profile") or []),
                seed=args.seed,
                adaptation_count=args.adaptation_per_user,
                selection_count=args.selection_per_user,
            )
        except ValueError as error:
            skipped.append({"user_id": user, "reason": str(error)})
            continue
        adaptation.extend(adapt_rows)
        selection.extend(select_rows)
        manifest.append(user_manifest)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "profile_adaptation.jsonl", adaptation)
    write_jsonl(args.output / "profile_selection.jsonl", selection)
    # Keep the official test file separate; downstream evolution receives only
    # profile_adaptation.jsonl, while the final scorer receives test.jsonl.
    write_jsonl(args.output / "test.jsonl", test_rows)
    write_jsonl(args.output / "users.jsonl", manifest)
    (args.output / "skipped_users.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2))
    stats = {
        "protocol": "strict_user_test_time_profile_only_self_supervised",
        "source_test": str(args.source_test),
        "seed": args.seed,
        "users_in_official_test": len(by_user),
        "users_adapted": len(manifest),
        "users_skipped": len(skipped),
        "rows": {
            "profile_adaptation": len(adaptation),
            "profile_selection": len(selection),
            "official_test": len(test_rows),
        },
        "adaptation_per_user": args.adaptation_per_user,
        "selection_per_user": args.selection_per_user,
        "official_test_target_used_during_adaptation": False,
        "pseudo_labels_are_historical_profile_abstracts": True,
    }
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
