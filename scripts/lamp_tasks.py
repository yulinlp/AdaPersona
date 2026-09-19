"""Original LaMP user-split adapters for tasks 2, 3, 4 and 5.

The native LaMP release stores one current task and a user's historical
profile in ``*_questions.json`` and the current gold in ``*_outputs.json``.
This module turns the historical profile items into deterministic,
leave-one-out adaptation rows.  The current test gold is kept only in the
prepared test file and is never passed to the evolver or runtime harness.

The adapter deliberately supports only LaMP-2/3/4/5.  LaMP-1/6/7 do not
provide a complete historical input-target pair suitable for the same RSI
feedback protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

try:
    from . import evolution_metrics as text_metrics
except ImportError:  # pragma: no cover
    import evolution_metrics as text_metrics


TASKS = (2, 3, 4, 5)
TASK_NAMES = {
    2: "LaMP-2 personalized topic classification",
    3: "LaMP-3 personalized product rating",
    4: "LaMP-4 personalized news headline generation",
    5: "LaMP-5 personalized scholarly title generation",
}


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _tokens(text: Any) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "").casefold())


def _label(text: Any) -> str:
    value = _clean(text).strip(" `\"'.,:;()[]{}")
    return value.casefold()


def _number(text: Any) -> int | None:
    match = re.fullmatch(r"\s*([1-5])\s*", str(text or ""))
    return int(match.group(1)) if match else None


def _task2_input(item: dict[str, Any]) -> str:
    # This is the category vocabulary used by the native LaMP-2 release.  It
    # is part of the task contract, not a heuristic inferred from a profile.
    categories = (
        "women, religion, politics, style & beauty, entertainment, culture & arts, "
        "sports, science & technology, travel, business, crime, education, "
        "healthy living, parents, food & drink"
    )
    text = _clean(item.get("text"))
    return (
        "Which category does this article relate to among the following categories? "
        "Just answer with the category name without further explanation.\n"
        f"categories: [{categories}] article: {text}"
    )


def _task3_input(item: dict[str, Any]) -> str:
    return (
        "What is the score of the following review on a scale of 1 to 5? "
        "Just answer with 1, 2, 3, 4, or 5 without further explanation.\n"
        f"review: {_clean(item.get('text'))}"
    )


def _task4_input(item: dict[str, Any]) -> str:
    return f"Generate a headline for the following article:\n{_clean(item.get('text'))}"


def _task5_input(item: dict[str, Any]) -> str:
    return f"Generate a title for the following abstract of a paper:\n{_clean(item.get('abstract'))}"


def history_task(task: int, item: dict[str, Any]) -> tuple[str, str] | None:
    """Return a historical input-target pair if the task has one."""
    task = int(task)
    if task == 2:
        target = _clean(item.get("category"))
        return (_task2_input(item), target) if target else None
    if task == 3:
        target = _clean(item.get("score"))
        return (_task3_input(item), target) if target else None
    if task == 4:
        target = _clean(item.get("title"))
        return (_task4_input(item), target) if target else None
    if task == 5:
        target = _clean(item.get("title"))
        return (_task5_input(item), target) if target else None
    raise ValueError(f"LaMP task {task} is not supported; choose one of {TASKS}")


def _spread_indices(size: int, count: int) -> list[int]:
    if count < 1 or size < 1:
        return []
    if size <= count:
        return list(range(size))
    if count == 1:
        return [size // 2]
    values = []
    seen = set()
    for index in range(count):
        value = round(index * (size - 1) / (count - 1))
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_native(source_root: Path, task: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    task_root = source_root / f"LaMP_{task}" / "test"
    questions_path = task_root / "test_questions.json"
    outputs_path = task_root / "test_outputs.json"
    if not questions_path.is_file() or not outputs_path.is_file():
        raise FileNotFoundError(f"Missing LaMP-{task} user test files under {task_root}")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
    golds = {str(item["id"]): str(item.get("output", "")) for item in outputs.get("golds", [])}
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"LaMP-{task} questions are empty or malformed")
    missing = [str(row.get("id")) for row in questions if str(row.get("id")) not in golds]
    if missing:
        raise ValueError(f"LaMP-{task} missing test golds for {len(missing)} rows")
    return questions, golds


def prepare_user_test(
    *, source_root: Path, output: Path, task: int, adaptation_items: int = 8,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Prepare strict profile-only test-time adaptation for one LaMP task."""
    if int(task) not in TASKS:
        raise ValueError(f"Only LaMP tasks {TASKS} are supported")
    if adaptation_items < 1:
        raise ValueError("adaptation_items must be positive")
    questions, golds = _load_native(source_root, int(task))
    if max_users is not None:
        questions = questions[: max(0, int(max_users))]
    adaptation: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for native in questions:
        native_id = str(native["id"])
        user_id = f"lamp{task}:{native_id}"
        profile = [item for item in (native.get("profile") or []) if isinstance(item, dict)]
        test_rows.append({
            "user_id": user_id,
            "sample_id": f"test:{native_id}",
            "input": str(native.get("input", "")),
            "target": golds[native_id],
            "profile": profile,
            "source_split": "user_test",
            "benchmark": "LaMP",
            "task": int(task),
            "native_id": native_id,
        })
        usable = []
        for index, item in enumerate(profile):
            pair = history_task(int(task), item)
            if pair is not None:
                usable.append((index, item, pair[0], pair[1]))
        chosen = [usable[i] for i in _spread_indices(len(usable), adaptation_items)]
        if not chosen:
            skipped.append({"user_id": user_id, "reason": "no historical input-target pairs"})
            continue
        adaptation_ids = []
        for index, item, current_input, target in chosen:
            history = [value for pos, value in enumerate(profile) if pos != index]
            item_id = str(item.get("id", index))
            sample_id = f"profile_adaptation:{native_id}:{item_id}"
            adaptation.append({
                "user_id": user_id,
                "sample_id": sample_id,
                "input": current_input,
                "target": target,
                "profile": history,
                "source_split": "test_profile_adaptation",
                "origin": "historical_profile_leave_one_out",
                "benchmark": "LaMP",
                "task": int(task),
                "native_id": native_id,
                "held_out_profile_item": item_id,
            })
            adaptation_ids.append(sample_id)
        manifest.append({
            "user_id": user_id,
            "native_id": native_id,
            "task": int(task),
            "profile_size": len(profile),
            "usable_history": len(usable),
            "adaptation_ids": adaptation_ids,
            "official_test_id": f"test:{native_id}",
        })

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "profile_adaptation.jsonl", adaptation)
    _write_jsonl(output / "test.jsonl", test_rows)
    _write_jsonl(output / "users.jsonl", manifest)
    (output / "skipped_users.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profile_sizes = [len(row.get("profile") or []) for row in test_rows]
    stats = {
        "benchmark": "LaMP",
        "task": int(task),
        "task_name": TASK_NAMES[int(task)],
        "source_root": str(source_root),
        "protocol": "user_test_profile_only_historical_target_adaptation",
        "users_in_test": len(test_rows),
        "users_adapted": len(manifest),
        "users_skipped": len(skipped),
        "rows": {"profile_adaptation": len(adaptation), "official_test": len(test_rows)},
        "adaptation_items_requested": int(adaptation_items),
        "profile_size": {
            "mean": round(statistics.mean(profile_sizes), 3) if profile_sizes else 0.0,
            "median": statistics.median(profile_sizes) if profile_sizes else 0.0,
            "min": min(profile_sizes) if profile_sizes else 0,
            "max": max(profile_sizes) if profile_sizes else 0,
        },
        "official_test_target_used_during_adaptation": False,
        "historical_targets_are_profile_items": True,
    }
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _render_profile(item: dict[str, Any]) -> str:
    fields = []
    for key, value in item.items():
        if key == "id" or value in (None, ""):
            continue
        fields.append(f"{key}: {_clean(value)}")
    return " | ".join(fields)


LAMP_SEED_HARNESS_CODE = r'''"""Generic black-box seed for LaMP-2/3/4/5."""
from collections import Counter
import json
import math
import re


def _tokens(text):
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text).lower())


def _render(item):
    values = []
    for key, value in item.items():
        if key == "id" or value in (None, ""):
            continue
        text = " ".join(str(value).split())
        # Profiles in LaMP-3/4 can contain long reviews/articles.  Keep the
        # seed within the frozen answer model's context contract; evolved
        # harnesses can choose a different budget-aware representation.
        if len(text) > 900:
            text = text[:900].rsplit(" ", 1)[0] + "..."
        values.append("%s: %s" % (key, text))
    return " | ".join(values)


def _retrieve(profile, query, k=6):
    if not profile:
        return []
    query_terms = Counter(_tokens(query))
    docs = [_tokens(_render(item)) for item in profile]
    df = Counter(term for doc in docs for term in set(doc))
    scored = []
    for index, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if not tf[term]:
                continue
            score += math.log(1.0 + (len(docs) + 1.0) / (df[term] + 1.0)) * min(tf[term], 3)
        scored.append((score, index, profile[index]))
    scored.sort(key=lambda value: (-value[0], value[1]))
    return [item for _, _, item in scored[:max(1, int(k))]]


def run(row, qa):
    profile = list(row.get("profile") or [])
    current = " ".join(str(row.get("input", "")).split())
    if len(current) > 10000:
        current = current[:10000].rsplit(" ", 1)[0] + "..."
    evidence = "\n".join("[Historical example %d] %s" % (i + 1, _render(item))
                           for i, item in enumerate(_retrieve(profile, current, 6)))
    prompt = """Solve the current LaMP task exactly.
Return only the requested answer, with no explanation, preface, markdown, or
extra labels. Historical examples are soft evidence about this user's behavior;
use them only when relevant and never copy a historical answer blindly.

Historical profile examples:
%s

Current task:
%s""" % (evidence or "No relevant historical examples.", current)
    return qa.generate(
        prompt,
        system="You are a precise personalized assistant. Follow the current task's output format exactly.",
        max_tokens=128,
        temperature=0.0,
    )
'''


class LampTaskAdapter:
    """Task-aware objective and prompt contract for LaMP-2/3/4/5."""

    def __init__(self, task: int):
        self.task = int(task)
        if self.task not in TASKS:
            raise ValueError(f"Only LaMP tasks {TASKS} are supported")
        self.name = f"lamp_{self.task}"
        self.task_name = TASK_NAMES[self.task]
        self.seed_code = LAMP_SEED_HARNESS_CODE
        if self.task == 2:
            self.metric_names = ("accuracy", "label_f1")
            self.objective_weights = {"accuracy": 1.0}
            self.metric_protocol = "lamp-exact-label-v1"
        elif self.task == 3:
            self.metric_names = ("accuracy", "rating_closeness")
            self.objective_weights = {"rating_closeness": 1.0}
            self.metric_protocol = "lamp-rating-exact-and-closeness-v1"
        else:
            self.metric_names = ("rouge1", "rouge2", "rougeL", "bleu", "meteor")
            self.objective_weights = {
                "rouge1": 0.25, "rougeL": 0.35, "bleu": 0.20, "meteor": 0.20
            }
            self.metric_protocol = text_metrics.METRIC_PROTOCOL
        weights = ",".join(f"{key}={value}" for key, value in self.objective_weights.items())
        self.objective_protocol = f"lamp-{self.task}-weighted-v1:{weights}"
        self.task_context = self.evolution_context()
        self.evaluation_boundary = (
            "The official current test input and target are held out from the evolver; "
            "only historical profile items and their held-out historical targets are available "
            "for this profile-only adaptation run."
        )

    def preflight(self) -> dict[str, Any]:
        if self.task in (4, 5):
            text_metrics.preflight()
        return {
            "benchmark": "LaMP", "task": self.task, "task_name": self.task_name,
            "metric_protocol": self.metric_protocol,
            "objective_protocol": self.objective_protocol,
            "objective_weights": self.objective_weights,
        }

    def evolution_context(self) -> str:
        if self.task == 2:
            output = "one category name from the task's supplied category list"
            history = "Historical profile items contain article text/title and a category label."
        elif self.task == 3:
            output = "one integer rating from 1 to 5"
            history = "Historical profile items contain a review text and the user's rating."
        elif self.task == 4:
            output = "one concise news headline"
            history = "Historical profile items contain an article and the user's historical headline."
        else:
            output = "one scholarly paper title"
            history = "Historical profile items contain an abstract and the user's historical title."
        return (
            f"Benchmark: original LaMP user-based task {self.task} ({self.task_name}).\n"
            f"The harness receives row['input'] and task-specific dictionaries in row['profile']. "
            f"The output contract is {output}; do not emit explanations. {history}\n"
            "The profile-only adaptation rows hold out one historical item from the runtime profile. "
            "The official current test target is not available during evolution."
        )

    @property
    def objective_description(self) -> str:
        terms = " + ".join(f"{value:.2f}*{key}" for key, value in self.objective_weights.items())
        return f"maximize this user's LaMP-{self.task} weighted score (0-100): {terms}"

    def row_metrics(self, row: dict[str, Any]) -> dict[str, float]:
        prediction = str(row.get("prediction", ""))
        target = str(row.get("target", ""))
        if self.task == 2:
            gold = _label(target)
            pred = _label(prediction)
            exact = 1.0 if pred == gold else 0.0
            # Partial label overlap is only a tie-breaker/diagnostic; the
            # exact category accuracy remains the main metric.
            gold_words, pred_words = set(_tokens(gold)), set(_tokens(pred))
            overlap = (2 * len(gold_words & pred_words) / max(1, len(gold_words) + len(pred_words)))
            return {"accuracy": exact, "label_f1": overlap, "prediction_tokens": float(len(_tokens(prediction)))}
        if self.task == 3:
            gold = _number(target)
            pred = _number(prediction)
            exact = 1.0 if pred is not None and gold is not None and pred == gold else 0.0
            closeness = 0.0 if pred is None or gold is None else max(0.0, 1.0 - abs(pred - gold) / 4.0)
            return {"accuracy": exact, "rating_closeness": closeness,
                    "prediction_tokens": float(len(_tokens(prediction)))}
        values = text_metrics.row_metrics({"prediction": prediction, "target": target})
        return values

    def weighted_score(self, metrics: dict[str, float]) -> float:
        # Accept both one-row metrics (``accuracy``) and an aggregate summary
        # (``mean_accuracy``).  The latter is what candidate selection uses.
        return sum(
            self.objective_weights[name] * float(
                metrics.get(name, metrics.get(f"mean_{name}", 0.0))
            )
            for name in self.objective_weights
        )

    def score_key(self, summary: dict[str, Any]) -> tuple[float, ...]:
        return tuple([self.weighted_score(summary)] + [float(summary.get(f"mean_{name}", 0.0))
                                                        for name in self.metric_names])

    def wins(self, child: dict[str, Any], parent: dict[str, Any]) -> bool:
        return child.get("errors", 0) == 0 and self.weighted_score(child) > self.weighted_score(parent) + 1e-12

    def summarize_rows(self, rows: list[dict[str, Any]], trace_limit: int = 20) -> dict[str, Any]:
        values = [self.row_metrics(row) for row in rows]
        result: dict[str, Any] = {
            "n": len(rows), "users": len({str(row.get("user_id", "")) for row in rows}),
            "errors": sum(bool(row.get("error")) for row in rows),
            "benchmark": "LaMP", "task": self.task, "task_name": self.task_name,
            "metric_protocol": self.metric_protocol,
            "objective_protocol": self.objective_protocol,
            "objective_weights": dict(self.objective_weights),
        }
        for name in self.metric_names:
            mean = sum(float(value.get(name, 0.0)) for value in values) / max(1, len(values))
            result[f"mean_{name}"] = mean
            result[f"mean_{name}_100"] = round(100.0 * mean, 3)
        result["weighted_score"] = self.weighted_score(result)
        result["weighted_score_100"] = round(100.0 * result["weighted_score"], 3)
        result['mean_qa_calls'] = sum(float(r.get('qa_calls', 0)) for r in rows) / max(1, len(rows))
        if self.task == 2:
            labels = sorted({_label(r.get('target', '')) for r in rows} |
                            {_label(r.get('prediction', '')) for r in rows})
            f1s = []
            for label in labels:
                tp = sum(_label(r.get('target', '')) == label == _label(r.get('prediction', '')) for r in rows)
                fp = sum(_label(r.get('prediction', '')) == label != _label(r.get('target', '')) for r in rows)
                fn = sum(_label(r.get('target', '')) == label != _label(r.get('prediction', '')) for r in rows)
                f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
            result['macro_f1'] = sum(f1s) / max(1, len(f1s))
        if self.task == 3:
            errors = [abs(_number(r['prediction']) - _number(r['target']))
                      if _number(r.get('prediction', '')) is not None else 4.0 for r in rows]
            result['mae'] = sum(errors) / max(1, len(errors))
            result['rmse'] = (sum(e * e for e in errors) / max(1, len(errors))) ** .5
            result['invalid_predictions'] = sum(_number(r.get('prediction', '')) is None for r in rows)
        traces = []
        for row, metrics in sorted(zip(rows, values), key=lambda pair: self.weighted_score(pair[1]))[:max(0, trace_limit)]:
            traces.append({
                "user_id": str(row.get("user_id", "")),
                "sample_id": str(row.get("sample_id", "")),
                "input": str(row.get("input", "")),
                "target": str(row.get("target", "")),
                "prediction": str(row.get("prediction", "")),
                "profile_size": len(row.get("profile") or []),
                "scores_100": {name: round(100.0 * float(metrics.get(name, 0.0)), 3)
                               for name in self.metric_names},
                "error": row.get("error", ""),
            })
        result["worst_traces"] = traces
        return result

    def build_failure_entries(self, parent_rows, child_rows, *, iteration, operation,
                              candidate_id, max_entries=80):
        parents = {(str(row.get("user_id", "")), str(row.get("sample_id", ""))): row
                   for row in parent_rows}
        entries = []
        for row in child_rows[:max_entries]:
            key = (str(row.get("user_id", "")), str(row.get("sample_id", "")))
            parent = parents.get(key, {})
            metrics = self.row_metrics(row)
            parent_metrics = self.row_metrics(parent) if parent else {}
            score = self.weighted_score(metrics)
            parent_score = self.weighted_score(parent_metrics) if parent else None
            delta = score - parent_score if parent_score is not None else None
            reasons = (["execution_error"] if row.get("error") else
                       ["regression_vs_parent"] if delta is not None and delta < 0 else
                       ["improvement"] if delta is not None and delta > 0 else ["training_observation"])
            profile = [item for item in row.get("profile", []) if isinstance(item, dict)]
            indices = sorted(set([0, len(profile) // 2, len(profile) - 1])) if profile else []
            trace = {
                "user_id": str(row.get("user_id", "")),
                "sample_id": str(row.get("sample_id", "")),
                "input": str(row.get("input", "")),
                "prediction": str(row.get("prediction", "")),
                "profile_size": len(profile),
                "scores_100": {name: round(100.0 * float(metrics.get(name, 0.0)), 3)
                               for name in self.metric_names},
                "error": row.get("error", ""),
            }
            entries.append({
                **trace, "iteration": iteration, "operation": operation,
                "candidate_id": candidate_id, "failure_types": reasons,
                "parent_prediction": parent.get("prediction", ""),
                "weighted_score": score, "weighted_score_100": round(100.0 * score, 3),
                "parent_weighted_score": parent_score,
                "delta_weighted_score": delta,
                "delta_weighted_points": round(100.0 * delta, 3) if delta is not None else None,
                "profile_examples": [profile[index] for index in indices],
                "profile_examples_selection": "first/middle/last, not a relevance ranking",
            })
        return entries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare profile-only LaMP user adaptation files.")
    parser.add_argument("--source-root", type=Path, default=Path("data/benchmarks/LaMP/user"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, choices=TASKS, required=True)
    parser.add_argument("--adaptation-items", type=int, default=8)
    parser.add_argument("--max-users", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = prepare_user_test(
        source_root=args.source_root, output=args.output, task=args.task,
        adaptation_items=args.adaptation_items, max_users=args.max_users,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
