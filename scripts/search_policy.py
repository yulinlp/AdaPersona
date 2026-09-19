"""Parent selection for tree/beam evolutionary harness search.

The original per-user runner is deliberately greedy: one confirmed incumbent
continues to the next operation.  This module keeps the evaluator unchanged
but provides a bounded, reproducible population policy for the next protocol:

* every recent candidate and globally strong candidate remains a possible
  parent within the bounded archive;
* a Pareto front over the four quality metrics is protected;
* parent slots mix elite, novel, under-explored, and island-diverse nodes;
* the final incumbent can still be selected by the scalar weighted objective.

The module is pure Python and does not call an LLM.  It only reads runner-owned
candidate artifacts, so it is safe to use from the orchestration process.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable


QUALITY_KEYS = ("mean_rouge1", "mean_rougeL", "mean_bleu", "mean_meteor")


def weighted_score(summary: dict[str, Any]) -> float:
    """Use the same scalar objective as the v3 evaluator, with replay fallback."""
    if "weighted_score" in summary:
        return float(summary.get("weighted_score", 0.0))
    return (
        0.25 * float(summary.get("mean_rouge1", 0.0))
        + 0.35 * float(summary.get("mean_rougeL", 0.0))
        + 0.20 * float(summary.get("mean_bleu", 0.0))
        + 0.20 * float(summary.get("mean_meteor", 0.0))
    )


def quality_vector(summary: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(summary.get(key, 0.0)) for key in QUALITY_KEYS)


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether left is no worse in every quality metric and better in one."""
    a, b = quality_vector(left), quality_vector(right)
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(nodes)
    return [node for node in values if not any(
        other is not node and dominates(other["summary"], node["summary"])
        for other in values
    )]


def _ast_features(code: str) -> tuple[float, ...]:
    """Small structural signature used only for diversity, never as fitness."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (1.0,)
    counts = {
        "functions": 0, "classes": 0, "calls": 0, "ifs": 0,
        "loops": 0, "imports": 0, "strings": 0, "nodes": 0,
    }
    for node in ast.walk(tree):
        counts["nodes"] += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts["functions"] += 1
        elif isinstance(node, ast.ClassDef):
            counts["classes"] += 1
        elif isinstance(node, ast.Call):
            counts["calls"] += 1
        elif isinstance(node, ast.If):
            counts["ifs"] += 1
        elif isinstance(node, (ast.For, ast.While, ast.comprehension)):
            counts["loops"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            counts["imports"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            counts["strings"] += 1
    # Log scaling keeps a large generated prompt from dominating distance.
    return tuple(math.log1p(value) for value in counts.values())


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = left["features"], right["features"]
    length = max(len(a), len(b))
    structural = sum(abs((a[i] if i < len(a) else 0.0) -
                         (b[i] if i < len(b) else 0.0)) for i in range(length))
    # AST distance is cheap and interpretable; the hash term prevents identical
    # feature counts from making semantically unrelated programs look identical.
    return structural + (0.0 if left["fingerprint"] == right["fingerprint"] else 1.0)


def _fingerprint(code: str) -> str:
    try:
        tree = ast.parse(code)
        return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()[:16]
    except SyntaxError:
        return hashlib.sha256(code.encode()).hexdigest()[:16]


def _safe_summary(event: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "n", "users", "errors", "mean_rouge1", "mean_rouge2", "mean_rougeL",
        "mean_bleu", "mean_meteor", "mean_title_coverage", "mean_qa_calls",
        "weighted_score", "weighted_score_100", "objective_protocol", "objective_weights",
    )
    return {key: event[key] for key in keys if key in event}


def _node_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    path_value = event.get("code_path")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    try:
        code = path.read_text()
    except OSError:
        return None
    summary = _safe_summary(event)
    if not summary or int(summary.get("errors", 0)):
        return None
    return {
        "candidate_id": str(event.get("candidate_id", path.stem)),
        "code_path": str(path),
        "predictions_path": str(event.get("predictions_path", "")),
        "summary": summary,
        "code": code,
        "fingerprint": _fingerprint(code),
        "features": _ast_features(code),
        "parent_code_path": event.get("parent_code_path") or event.get("exploration_parent"),
        "parent_candidate_id": event.get("parent_candidate_id"),
        "island_id": event.get("island_id"),
    }


def _incumbent_node(incumbent: dict[str, Any]) -> dict[str, Any]:
    code = Path(incumbent["code_path"]).read_text()
    return {
        "candidate_id": str(incumbent.get("candidate_id", "incumbent")),
        "code_path": str(incumbent["code_path"]),
        "predictions_path": str(incumbent["predictions_path"]),
        "summary": incumbent["summary"],
        "code": code,
        "fingerprint": _fingerprint(code),
        "features": _ast_features(code),
        "parent_code_path": None,
        "parent_candidate_id": None,
        "island_id": incumbent.get("island_id"),
        "is_incumbent": True,
    }


def _deduplicate(nodes: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for node in nodes:
        key = node["fingerprint"]
        if key in seen:
            continue
        seen.add(key)
        result.append(node)
    return result


def _island(node: dict[str, Any], count: int) -> int:
    if node.get("island_id") is not None:
        return int(node["island_id"]) % max(1, count)
    return int(node["fingerprint"][:8], 16) % max(1, count)


def select_parent_branches(
    *,
    incumbent: dict[str, Any],
    archive: list[dict[str, Any]],
    branch_count: int,
    beam_width: int,
    archive_size: int,
    island_count: int,
    iteration: int,
    operation: str,
) -> list[tuple[dict[str, Any], int]]:
    """Select diverse parent nodes and allocate exactly ``branch_count`` slots.

    The returned shape matches the existing runner's proposal loop.  The
    incumbent is deliberately reserved at least half of the proposal slots
    (two of four in the normal setting); the remaining slots explore archive
    lineages.  This keeps early search from spending the whole budget repairing
    a low-scoring branch while still preserving tree-search diversity.
    """
    incumbent_node = _incumbent_node(incumbent)
    nodes = [incumbent_node]
    # A bounded archive should not mean "last N only": a strong early branch
    # can be exactly the escape route needed after a locally attractive turn.
    # Merge the recent window with the global top-scoring window, then
    # de-duplicate by candidate identity before reading code artifacts.
    limit = max(1, int(archive_size))
    recent_events = list(archive[-limit:])
    ranked_events = sorted(
        archive,
        key=lambda event: (
            weighted_score(_safe_summary(event)),
            float(event.get("mean_rougeL", 0.0)),
        ),
        reverse=True,
    )[:limit]
    event_pool = []
    seen_events = set()
    for event in [*recent_events, *ranked_events]:
        identity = (event.get("candidate_id"), event.get("code_path"))
        if identity in seen_events:
            continue
        seen_events.add(identity)
        event_pool.append(event)
    for event in event_pool:
        node = _node_from_event(event)
        if node is not None:
            nodes.append(node)
    nodes = _deduplicate(nodes)
    if not nodes:
        return [(incumbent, max(1, int(branch_count)))]

    # Protect the best scalar solution and the current Pareto front.  The
    # remainder is available for structural/island exploration.
    front = pareto_front(nodes)
    elite = max(nodes, key=lambda n: (weighted_score(n["summary"]),
                                      float(n["summary"].get("mean_rougeL", 0.0))))
    candidates = _deduplicate([elite, incumbent_node, *front, *nodes])
    selected: list[dict[str, Any]] = []

    def add(node):
        if node["fingerprint"] not in {item["fingerprint"] for item in selected}:
            selected.append(node)

    add(elite)
    # One representative per island encourages independent lineages.  This is
    # island-style diversity, while the archive remains shared globally.
    for island_id in range(max(1, int(island_count))):
        island_nodes = [n for n in candidates if _island(n, island_count) == island_id]
        if island_nodes:
            add(max(island_nodes, key=lambda n: weighted_score(n["summary"])))
        if len(selected) >= max(1, int(beam_width)):
            break

    # Fill remaining beam slots by novelty plus a small under-exploration bonus.
    child_counts = {}
    for event in archive:
        parent = event.get("parent_code_path") or event.get("exploration_parent")
        if parent:
            child_counts[str(parent)] = child_counts.get(str(parent), 0) + 1
    while len(selected) < max(1, int(beam_width)) and len(selected) < len(candidates):
        remaining = [n for n in candidates if n["fingerprint"] not in {x["fingerprint"] for x in selected}]
        chosen = max(remaining, key=lambda n: (
            min(_distance(n, s) for s in selected) if selected else 0.0,
            -child_counts.get(n["code_path"], 0),
            weighted_score(n["summary"]),
        ))
        add(chosen)

    selected = selected[:max(1, int(beam_width))]
    # Keep the incumbent allocation explicit.  Rotation is applied only to
    # exploratory nodes, otherwise a full beam can accidentally omit the
    # incumbent after the archive has grown.
    incumbent_fingerprint = incumbent_node["fingerprint"]
    explorers = [node for node in selected
                 if node["fingerprint"] != incumbent_fingerprint]
    if explorers:
        offset = (int(iteration) + (0 if operation == "macro_strategy" else 1)) % len(explorers)
        explorers = explorers[offset:] + explorers[:offset]

    total = max(1, int(branch_count))
    incumbent_slots = max(1, (total + 1) // 2)
    incumbent_slots = min(total, incumbent_slots)
    exploration_slots = total - incumbent_slots
    allocations = [[incumbent_node, incumbent_slots]]
    if explorers:
        for index in range(exploration_slots):
            node = explorers[index % len(explorers)]
            # Keep one tuple per lineage where possible.  If there is only one
            # archive node, assigning it multiple slots still gives the
            # operation-level planner distinct hypotheses.
            for item in allocations:
                if item[0]["fingerprint"] == node["fingerprint"]:
                    item[1] += 1
                    break
            else:
                allocations.append([node, 1])
    else:
        # At startup the archive has no usable alternatives: all slots should
        # stay on the seed/incumbent while the planner creates diverse ideas.
        allocations[0][1] += exploration_slots
    return [(node, count) for node, count in allocations]
