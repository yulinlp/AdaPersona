"""Shared v2 proposal, training-feedback and isolated-evaluation helpers.

This is the first black-box evolution protocol for the project.  It replaces
the old fixed E/A/I/R/O/G search with one executable program.  The only
contract is ``harness.py::run(row, qa) -> str``; the program may implement any
combination of memory, retrieval, graph construction, prompting, candidate
selection, and multi-call refinement that can be expressed through an
ordinary vLLM chat-completions API.

Each evolution round chooses one of two operation families:

1. ``macro_strategy``: replace or substantially restructure the main
   personalization strategy;
2. ``micro_repair``: preserve the current strategy and repair an observed
   failure in its internal implementation.

For each operation the Qwen3.8-27B evolver may propose several code branches.
The branch with the best training-set objective wins greedily; the parent is a
valid no-op if every proposal is worse.  Validation/test outputs are never
used by the evolver or the acceptance decision in the default fit mode.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

try:
    from .evolution_metrics import (
        row_metrics, METRIC_PROTOCOL, OBJECTIVE_PROTOCOL, OBJECTIVE_WEIGHTS,
        weighted_metric_score,
    )
    from .isolated_runtime import run_isolated
    from .blackbox_harness import validate_harness_code
except ImportError:
    from evolution_metrics import (
        row_metrics, METRIC_PROTOCOL, OBJECTIVE_PROTOCOL, OBJECTIVE_WEIGHTS,
        weighted_metric_score,
    )
    from isolated_runtime import run_isolated
    from blackbox_harness import validate_harness_code

try:
    from blackbox_harness import (
        CompiledHarness,
        HarnessValidationError,
        SEED_HARNESS_CODE,
        VLLMQA,
        code_fingerprint,
        runtime_row,
        save_code,
        strip_code_fence,
    )
    from longlamp_rsi import (
        VLLMOpenAIClient,
        extract_json,
        load_jsonl,
        task_title,
        title_coverage,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style invocation
    from scripts.blackbox_harness import (
        CompiledHarness,
        HarnessValidationError,
        SEED_HARNESS_CODE,
        VLLMQA,
        code_fingerprint,
        runtime_row,
        save_code,
        strip_code_fence,
    )
    from scripts.longlamp_rsi import (
        VLLMOpenAIClient,
        extract_json,
        load_jsonl,
        task_title,
        title_coverage,
        write_jsonl,
    )


OPERATION_NAMES = ("macro_strategy", "micro_repair")


def write_jsonl(path, rows):
    """Atomic per-user checkpoints; a killed runner cannot leave a half archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temp.replace(path)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def _score100(value: float) -> float:
    return round(100.0 * float(value), 3)


def weighted_score(summary: dict[str, Any]) -> float:
    """Return the [0, 1] aggregate objective used for candidate selection."""
    return sum(float(OBJECTIVE_WEIGHTS[name]) * float(summary.get('mean_' + name, 0.0))
               for name in OBJECTIVE_WEIGHTS)


def score_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Weighted objective first, then deterministic diagnostic tie-breakers."""
    return (
        weighted_score(summary),
        float(summary.get('mean_rougeL', 0.0)),
        float(summary.get('mean_rouge1', 0.0)),
        float(summary.get('mean_bleu', 0.0)),
        float(summary.get('mean_meteor', 0.0)),
        float(summary.get('mean_title_coverage', 0.0)),
    )


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row.get("user_id", ""))].append(row)
    for user in result:
        result[user].sort(key=lambda row: str(row.get("sample_id", "")))
    return dict(result)


def select_rows(rows: list[dict[str, Any]], max_users: int | None, per_user: int | None) -> list[dict[str, Any]]:
    by_user = grouped(rows)
    users = sorted(by_user)
    if max_users is not None:
        users = users[: max(0, int(max_users))]
    selected: list[dict[str, Any]] = []
    for user in users:
        values = by_user[user]
        if per_user is not None:
            values = values[: max(0, int(per_user))]
        selected.extend(values)
    return selected


def _trace(row: dict[str, Any], *, include_qa_trace: bool = False) -> dict[str, Any]:
    metrics = row_metrics(row)
    result = {
        "user_id": str(row.get("user_id", "")), "sample_id": str(row.get("sample_id", "")),
        "input": str(row.get("input", "")), "target": str(row.get("target", "")),
        "prediction": str(row.get("prediction", "")), "profile_size": len(row.get("profile") or []),
        "scores_100": {k: _score100(v) for k, v in metrics.items() if k != "prediction_tokens"},
        "output_words": metrics["prediction_tokens"], "qa_calls": row.get("qa_calls", 0),
        "error": row.get("error", ""),
    }
    if include_qa_trace:
        result["qa_trace"] = row.get("qa_trace", [])
    return result


def summarize_rows(rows: list[dict[str, Any]], trace_limit: int = 20) -> dict[str, Any]:
    metrics = [row_metrics(row) for row in rows]
    result = {"n": len(rows), "users": len({str(r.get("user_id", "")) for r in rows}),
              "errors": sum(bool(r.get("error")) for r in rows)}
    for name in ("rouge1", "rouge2", "rougeL", "bleu", "meteor"):
        result["mean_" + name] = _mean(m[name] for m in metrics)
        result["mean_" + name + "_100"] = _score100(result["mean_" + name])
    result['weighted_score'] = weighted_score(result)
    result['weighted_score_100'] = _score100(result['weighted_score'])
    result['objective_protocol'] = OBJECTIVE_PROTOCOL
    result['objective_weights'] = dict(OBJECTIVE_WEIGHTS)
    result["mean_title_coverage"] = _mean(title_coverage(r.get("input", ""), r.get("prediction", "")) for r in rows)
    result["mean_title_coverage_100"] = _score100(result["mean_title_coverage"])
    result["mean_qa_calls"] = _mean(r.get("qa_calls", 0) for r in rows)
    # All eight training examples are visible; ordering follows the actual objective.
    ordered = sorted(zip(rows, metrics), key=lambda pair: (not bool(pair[0].get("error")), pair[1]["rougeL"]))
    result["worst_traces"] = [_trace(r) for r, _ in ordered[:max(0, trace_limit)]] if trace_limit else []
    result["metric_protocol"] = METRIC_PROTOCOL
    return result


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("user_id", "")), str(row.get("sample_id", ""))


def build_failure_entries(parent_rows, child_rows, *, iteration, operation, candidate_id, max_entries=80):
    """Training-only evidence, including successes and full reference comparisons."""
    parents = {_pair_key(row): row for row in parent_rows}
    entries = []
    for row in child_rows:
        parent = parents.get(_pair_key(row), {})
        metric, pm = row_metrics(row), row_metrics(parent)
        delta = metric["rougeL"] - pm["rougeL"]
        weighted = weighted_metric_score(metric)
        parent_weighted = weighted_metric_score(pm) if parent else None
        weighted_delta = weighted - parent_weighted if parent else None
        reasons = (["execution_error"] if row.get("error") else
                   ["regression_vs_parent"] if parent and weighted_delta < 0 else
                   ["improvement"] if parent and weighted_delta > 0 else ["training_observation"])
        profile = [p for p in row.get("profile", []) if isinstance(p, dict)]
        # Do not silently assume the source ordering represents recency.
        indices = sorted(set([0, len(profile)//2, len(profile)-1])) if profile else []
        trace = _trace(row, include_qa_trace=True)
        # Keep the bank useful across steps without duplicating full reference
        # answers in every historical entry.  Current-candidate comparisons
        # still expose the reference to the evolver; the persistent bank keeps
        # scores, predictions, traces, and failure causes only.
        trace.pop('target', None)
        entries.append({
            **trace,
            "iteration": iteration, "operation": operation, "candidate_id": candidate_id,
            "failure_types": reasons, "parent_prediction": parent.get("prediction", ""),
            "rougeL": metric["rougeL"], "parent_rougeL": pm["rougeL"] if parent else None,
            "delta_rougeL": delta if parent else None,
            "weighted_score": weighted, "weighted_score_100": _score100(weighted),
            "parent_weighted_score": parent_weighted,
            "delta_weighted_score": weighted_delta,
            "delta_weighted_points": _score100(weighted_delta) if weighted_delta is not None else None,
            "profile_examples": [profile[i] for i in indices],
            "profile_examples_selection": "first/middle/last, not a relevance ranking",
        })
    return entries[:max(0, max_entries)]


class FailureBank:
    """Persistent TRAINING feedback; select across steps, candidates and samples."""
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []

    @classmethod
    def load(cls, path):
        bank = cls(path)
        if bank.path.exists():
            # Corruption must be visible; never silently lose a prior failure.
            bank.entries = load_jsonl(bank.path)
        return bank

    def add(self, entries):
        def key(entry):
            return (entry.get('candidate_id'), entry.get('sample_id'), entry.get('iteration'), entry.get('operation'))
        existing = {key(e) for e in self.entries}
        for entry in entries:
            if key(entry) in existing:
                continue
            self.entries.append(entry)
            existing.add(key(entry))
        write_jsonl(self.path, self.entries)

    def recent(self, limit=16):
        return self.entries[-limit:] if limit > 0 else []

    def prompt_view(self, limit=16):
        if limit <= 0:
            return []
        groups = {}
        for entry in self.entries:
            groups.setdefault(entry.get("candidate_id", "unknown"), []).append(entry)
        # Round-robin across the entire history, recent first, then older steps.
        keys = list(reversed(groups))
        if len(keys) > limit:
            keys = [keys[round(i * (len(keys)-1) / (limit-1))] for i in range(limit)] if limit > 1 else keys[:1]
        output = []
        depth = 0
        while len(output) < limit:
            added = False
            for key in keys:
                entries = sorted(groups[key], key=lambda e: (
                    "execution_error" not in e.get("failure_types", []),
                    e.get("weighted_score", e.get("rougeL", 0)),
                    str(e.get("sample_id", ""))))
                if depth >= len(entries):
                    continue
                item = dict(entries[depth])
                traces = [t for t in item.get("qa_trace", []) if "prompt" in t or "error" in t]
                # Keep whole first/final calls; explicitly report selection.
                item["qa_trace"] = (traces if len(traces) <= 2 else [traces[0], traces[-1]]) if not output else []
                item["qa_trace_total"] = len(traces)
                item['trace_selection'] = 'first and final on one representative record; other full traces remain in artifacts'
                output.append(item)
                added = True
                if len(output) == limit:
                    break
            if not added:
                break
            depth += 1
        return output


def _history_summary(history, limit=24):
    keys = ("event", "iteration", "operation", "candidate_id", "accepted", "accepted_candidate",
            "delta_rougeL", "delta_weighted_score", "parent_rougeL", "child_rougeL",
            "weighted_score_100", "hypothesis", "error", "exploration_parent")
    result = []
    for record in history[-limit:]:
        value = {k: record[k] for k in keys if k in record}
        # LaMP-2/3 use accuracy-oriented objectives.  Preserve all task
        # metric deltas without making the shared prompt know their names.
        for key, item in record.items():
            if (key.startswith(("mean_", "parent_", "child_", "delta_"))
                    and key not in value and isinstance(item, (int, float))):
                value[key] = item
        result.append(value)
    return result


def _archive_summary(archive, limit=8):
    """Send real source/diffs; the API evolver cannot open paths on disk."""
    import difflib
    if not archive:
        return []
    def rank_key(record):
        metric_values = [float(value) for key, value in record.items()
                         if key.startswith("mean_") and not key.endswith("_100")
                         and isinstance(value, (int, float))]
        return (bool(record.get("accepted")),
                float(record.get("weighted_score", -1)),
                max(metric_values, default=-1.0))

    ranked = sorted(archive, key=rank_key, reverse=True)
    chosen = []
    for item in [archive[-1], *ranked, *reversed(archive)]:
        if item.get("candidate_id") not in {x.get("candidate_id") for x in chosen}:
            chosen.append(item)
        if len(chosen) >= limit:
            break
    result = []
    for item in chosen:
        value = {k: v for k, v in item.items() if k not in ("code", "worst_traces")}
        source = item.get("code", "")
        # Only runner-written archive paths are read, never agent-supplied paths.
        if not source and item.get("code_path") and Path(item["code_path"]).is_file():
            source = Path(item["code_path"]).read_text()
        if item is chosen[0]:
            value["source"] = source
        else:
            value["source_diff_from_first_archive_item"] = "".join(difflib.unified_diff(
                result[0].get("source", "").splitlines(True), source.splitlines(True)))
        result.append(value)
    return result


def load_strategy_library(path: Path | None = None) -> str:
    path = path or Path(__file__).resolve().parents[1] / "docs" / "BLACKBOX_STRATEGY_LIBRARY.md"
    if path.exists():
        return path.read_text()
    return "Use arbitrary black-box memory, retrieval, prompting, candidate selection, and multi-call refinement; do not use logits or hidden states."


RUNTIME_CONTRACT = '''Only fixed interface: synchronous def run(row, qa) -> str.
row contains user_id, sample_id, input, profile; NEVER the current reference answer.
profile is a list of task-specific historical dictionaries; fields may include
titles, text, abstracts, labels, scores, dates, or other public profile metadata.
Each sample runs in a fresh process: classes/functions are supported; globals do not persist.
qa exposes ONLY:
  generate(prompt: str, *, system='You are a helpful assistant.', max_tokens=256,
           temperature=0.0, top_p=1.0) -> str
  generate_many(prompts: list[str], *, same keyword arguments) -> list[str]
  embed(texts: list[str], *, instruction=None) -> list[list[float]]
generate_many counts each prompt against the 8-generation-call budget. All runtime
generation, judging and summarizing MUST use this frozen Qwen2.5-7B-Instruct client.
The broker clamps temperature/top_p to deterministic temperature=0, top_p=1,
top_k=1, seed=0 during evaluation; do not rely on sampling noise for a gain.
Target context capacity is 8192 tokens INCLUDING output; choose prompts and max_tokens accordingly.
Embeddings are optional frozen Qwen3-Embedding-0.6B, normalized 1024D vectors, max 2048
tokens per text (longer texts truncate); instruction is optional and task-dependent.
Embedding budget: 4096 texts / 2,000,000 characters per sample; batch related texts.
Imports: numpy, scipy, sklearn, networkx, collections, functools, itertools,
json, math, re, statistics, string. Ordinary classes, comprehensions, exceptions work.
Import every module you use (for example import math); no modules are implicitly injected.
Object identity via id() is unavailable; deduplicate with content keys or profile indices.
Use deterministic ordering and tie-breaking; set order is not a semantic ranking.
Give randomized library algorithms explicit random_state=0 (or an equivalent fixed seed).
Do not put complete historical reference answers in source, including comments or examples.
Infer reusable transformations; retrieve examples from the supplied runtime profile instead.
Avoid fixing only the single worst example at the expense of other historical tasks.
No filesystem/network/subprocess access, downloads, introspection or model-weight training.
No Qwen3.8 calls from runtime. Do not access qa internals. Only in-memory computation
and the three public qa methods are available. 300s wall / 120s CPU / 8GiB address-space
limits per sample; return nonempty str. Runtime errors are returned as training feedback.
'''


def runtime_contract_for(model_id='Qwen2.5-7B-Instruct', context_tokens=8192):
    """Render the black-box contract for the selected frozen answer model."""
    contract = RUNTIME_CONTRACT.replace('Qwen2.5-7B-Instruct', str(model_id))
    contract = contract.replace(
        'Target context capacity is 8192 tokens',
        f'Target context capacity is {int(context_tokens)} tokens',
    )
    contract = contract.replace(
        'No Qwen3.8 calls from runtime. Do not access qa internals.',
        'Do not call any model except through the public qa interface. Do not access qa internals.',
    )
    return contract


def evolution_prompt(*, parent_code, operation, iteration, branch, current_summary,
                     failure_bank, history, archive, strategy_library,
                     agent_model='Qwen/Qwen3.8-27B',
                     runtime_model='Qwen2.5-7B-Instruct', runtime_context=8192,
                     task_context=None, objective_description=None,
                     evaluation_boundary=None):
    instruction = (
        "Explore a substantively different, evidence-supported hypothesis. You may replace the entire workflow; "
        "the parent is a comparison point, not a required template."
        if operation == "macro_strategy" else
        "Repair concrete weaknesses of the supplied branch. Preserve its core hypothesis unless the evidence "
        "shows it is untenable. This branch can be exploratory and below the incumbent score."
    )
    # The user identity is not an evolution signal.  Keep it in runner-owned
    # artifacts for isolation/audit, but remove it from the outer model's
    # evidence so the user's name cannot become an accidental prior.
    prompt_summary = dict(current_summary)
    prompt_summary['worst_traces'] = [
        {k: v for k, v in trace.items() if k != 'user_id'}
        for trace in current_summary.get('worst_traces', [])
    ]
    prompt_failures = [
        {k: v for k, v in item.items() if k != 'user_id'}
        for item in failure_bank.prompt_view(12)
    ]
    prompt_archive = [
        {k: v for k, v in item.items() if k != 'user_id'}
        for item in _archive_summary(archive, 3)
    ]
    task_context = task_context or (
        'Benchmark: LongLaMP abstract generation. The harness receives an abstract task and '
        'historical title/abstract dictionaries and must return only the current abstract.'
    )
    objective_description = objective_description or (
        'maximize this user\'s weighted training score (0-100): '
        '0.25*ROUGE-1 + 0.35*ROUGE-L + 0.20*BLEU + 0.20*METEOR'
    )
    evaluation_boundary = evaluation_boundary or (
        'No held-out targets are available. We are measuring training fitting, NOT generalization.'
    )
    return f'''You are {agent_model}, the training-time Python harness evolver for ONE isolated user.
Step {iteration}; operation={operation}; branch={branch}.
{instruction}

{runtime_contract_for(runtime_model, runtime_context)}

Task-specific contract and allowed historical evidence:
{task_context}

Primary objective: {objective_description}, with zero execution errors.
Treat execution logs as evidence: a zero-error evaluated parent is executable. Do not invent
syntax errors or pretend unavailable references/tools were observed. Distinguish hypotheses from facts.
Do not optimize title repetition, code length, complexity or number of tool calls for their own sake.
Training references below are available for diagnosis only. Infer reusable user-specific behavior;
never embed reference answers, task-to-answer tables, sample-ID lookup, or disguised answer memorization.
{evaluation_boundary}

The library is non-exhaustive and unranked. Any idea may be ignored or combined.
Simple approaches and novel approaches are equally valid. No retrieval, graph, embedding,
multi-call workflow or internal module is mandatory. Explain the hypothesis and evidence briefly
in source comments; implement it, do not merely describe it.

Optional reference strategies and capabilities:
{strategy_library}

Current branch evaluation; full input/reference/prediction comparisons:
{json.dumps(prompt_summary, ensure_ascii=False)}

Training feedback sampled ACROSS previous steps (includes successes and failures):
{json.dumps(prompt_failures, ensure_ascii=False)}

Evolution history (incumbent acceptance and exploratory branches are different):
{json.dumps(_history_summary(history), ensure_ascii=False)}

Historical alternatives with actual source/diffs, not just inaccessible paths:
{json.dumps(prompt_archive, ensure_ascii=False)}

Current branch source (replaceable):
```python
{parent_code}
```

Return ONLY complete Python source, optionally in one Python fence. No JSON artifact.
There is no minimum code length. Stop when the complete runnable program is finished.
'''


def agent_chat(
    api_url: str,
    model: str,
    prompt: str,
    *,
    timeout: float,
    retries: int,
    retry_wait: float,
    max_tokens: int | None,
    record=None,
) -> str:
    url = str(api_url).rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a code evolution agent. Output complete Python source only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                result = json.loads(response.read().decode("utf-8"))
            choices = result.get("choices") or []
            if record is not None:
                record({'request': payload, 'response': result, 'attempt': attempt + 1})
            if not choices:
                raise RuntimeError(f"agent returned no choices: {result}")
            if choices[0].get('finish_reason') == 'length':
                raise RuntimeError('Evolver output truncated by serving/context limit; see request artifact')
            content = (choices[0].get("message") or {}).get("content", "")
            if isinstance(content, list):
                content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
            if not str(content or '').strip():
                raise RuntimeError('Evolver emitted an empty answer; retry original task, not a context-free code repair')
            return str(content or "").strip()
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, IndexError, RuntimeError) as error:
            last_error = error
            if record is not None:
                record({'request': payload, 'error': str(error), 'attempt': attempt + 1})
            if attempt + 1 < max(1, int(retries)) and retry_wait > 0:
                time.sleep(float(retry_wait) * (attempt + 1))
    raise RuntimeError(f"agent request failed after {retries} attempts: {last_error}") from last_error


def fit_agent_context(prompt, api_url, record, model_id='Qwen/Qwen3.8-27B'):
    """Use serving tokenizer; keep full parent/current training comparisons intact."""
    root = api_url.rstrip('/').removesuffix('/v1')
    with urllib.request.urlopen(api_url.rstrip('/') + '/models', timeout=15) as response:
        models = json.load(response)['data']
    model = next(m for m in models if m['id'] == model_id)
    def count(text):
        request = urllib.request.Request(root + '/tokenize', data=json.dumps({
            'model': model['id'], 'prompt': text}).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)['count']
    capacity = int(model.get('max_model_len', 16384))
    small_context = capacity <= 16384
    if small_context:
        # The frozen 7B service is also a valid comparison Code Agent, but it
        # has an 8K context.  Keep the complete parent source and the current
        # measured evidence while replacing the long literature catalog with
        # its compact, actionable interface-level summary.
        library_start = prompt.find('Optional reference strategies and capabilities:')
        library_end = prompt.find('\n\nCurrent branch evaluation;', library_start)
        if library_start >= 0 and library_end > library_start:
            compact_library = '''Optional reference strategies and capabilities (compact 8K-agent view):
- history/profile retrieval: lexical, semantic, recency, diversity, graph-like summaries;
- prompt construction: task-conditioned profile/style cards and relevant examples;
- black-box generation: direct, plan-then-write, draft/revise, candidate selection;
- robust fallbacks: empty/irrelevant history, budget-aware calls, deterministic safe output.
Use these as hypotheses, not mandatory modules; implement and test one causal change.'''
            prompt = prompt[:library_start] + compact_library + prompt[library_end:]
    parent_marker = 'Current branch source (replaceable):'
    parent = prompt.split(parent_marker, 1)[-1] if parent_marker in prompt else ''
    fence = re.search(r'```python\s*(.*?)```', parent, re.S)
    if fence:
        parent = fence.group(1)
    # Reserve room for a complete source reply.  The evolver service now has a
    # 64K context, so keep the original parent-proportional headroom instead
    # of the temporary 2K--4K cap used while it was limited to 16K.
    # We do not send max_tokens; serving context remains the hard upper bound.
    parent_tokens = count(parent)
    reserve = (
        max(1024, int(parent_tokens * 1.05))
        if small_context else
        max(4096, int(parent_tokens * 1.3))
    )
    budget = capacity - reserve - 256  # chat-template overhead; no max_tokens in request
    original = prompt
    omitted = []
    tokens = count(prompt)
    # Reduce historical detail before selecting fewer full training comparisons.
    sections = [
        ('Training feedback sampled ACROSS previous steps', 'Evolution history ('),
        ('Historical alternatives with actual source/diffs', parent_marker),
        ('Evolution history (', 'Historical alternatives with actual source/diffs'),
    ]
    for start_marker, end_marker in sections:
        if tokens <= budget:
            break
        start, end = prompt.find(start_marker), prompt.find(end_marker)
        if start >= 0 and end > start:
            section = prompt[start:end]
            # Retain records across steps but discard bulky trace/profile/source copies.
            brace = section.find('\n')
            try:
                records = json.loads(section[brace:].strip())
                compact = [{k: v for k, v in item.items() if k not in
                            ('qa_trace', 'profile_examples', 'source', 'source_diff_from_first_archive_item',
                             'parent_prediction', 'prediction', 'target')} for item in records]
                replacement = section[:brace] + '\n' + json.dumps(compact, ensure_ascii=False) + '\n\n'
            except (ValueError, TypeError):
                replacement = start_marker + '\n[Omitted for serving context capacity.]\n\n'
            prompt = prompt[:start] + replacement + prompt[end:]
            omitted.append(start_marker)
            tokens = count(prompt)
    # A small deployment cannot always fit all eight comparisons plus growing code.
    # Keep every sample's scores, rotate whole examples (never partial references).
    start_marker, end_marker = 'Current branch evaluation;', 'Training feedback sampled ACROSS'
    start, end = prompt.find(start_marker), prompt.find(end_marker)
    if tokens > budget and start >= 0 and end > start:
        section = prompt[start:end]
        split = section.find('\n')
        summary = json.loads(section[split:].strip())
        cases = summary.get('worst_traces', [])
        branch = re.search(r'Step (\d+); operation=[^;]+; branch=(\d+)', prompt)
        offset = (int(branch.group(1)) + 2*int(branch.group(2))) if branch else 0
        original_cases = list(cases)
        for keep in (6, 4, 2, 1):
            if tokens <= budget or len(original_cases) <= keep:
                continue
            indices = [0]
            for i in range(len(original_cases)):
                index = (offset+i) % len(original_cases)
                if index not in indices:
                    indices.append(index)
                if len(indices) == keep:
                    break
            indices = indices[:keep]
            summary['worst_traces'] = [original_cases[i] for i in indices]
            summary['all_training_case_scores'] = [{k:v for k,v in case.items() if k not in
                ('input','target','prediction','qa_trace')} for case in original_cases]
            summary['context_selection'] = {
                'total_training_cases': len(original_cases), 'full_comparisons_included': keep,
                'included_sample_ids': [original_cases[i]['sample_id'] for i in indices],
                'omitted_sample_ids': [c['sample_id'] for i,c in enumerate(original_cases) if i not in indices],
                'rule': 'worst objective case plus step/branch rotation; full records preserved on disk'}
            end = prompt.find(end_marker)
            prompt = prompt[:start] + section[:split] + '\n' + json.dumps(summary, ensure_ascii=False) + '\n\n' + prompt[end:]
            tokens = count(prompt)
        omitted.append('Current full training comparisons selected by step/branch rotation; all case scores retained')
    # Extremely long historical hypotheses must not crowd out actual current evidence.
    if tokens > budget:
        for start_marker, end_marker in sections:
            start, end = prompt.find(start_marker), prompt.find(end_marker)
            if start < 0 or end <= start:
                continue
            section = prompt[start:end]
            split = section.find('\n')
            try:
                records = json.loads(section[split:].strip())
                keys = {'candidate_id','iteration','operation','accepted','accepted_candidate',
                        'sample_id','failure_types','scores_100','delta_rougeL','error','event',
                        'mean_rougeL_100','child_rougeL','parent_rougeL',
                        'weighted_score_100','delta_weighted_score'}
                compact = []
                for item in records:
                    compact.append({
                        k: v for k, v in item.items()
                        if k in keys or k.startswith(("mean_", "parent_", "child_", "delta_"))
                    })
                prompt = prompt[:start] + section[:split] + '\n' + json.dumps(compact, ensure_ascii=False) + '\n\n' + prompt[end:]
                omitted.append('Historical metadata-only: ' + start_marker)
                tokens = count(prompt)
            except (ValueError, TypeError):
                pass
            if tokens <= budget:
                break
    record({'phase': 'context_preflight', 'input_tokens': tokens, 'capacity': capacity,
            'output_space_reserved': reserve, 'compacted_sections': omitted,
            'original_prompt': original if omitted else None})
    if tokens > budget:
        raise RuntimeError(f'Context capacity insufficient: input {tokens}, reserve {reserve}, capacity {capacity}; '
                           'full parent preserved and whole training examples selected; increase evolver context')
    return prompt


def parse_proposal(text: str, expected_operation: str, parent_code: str) -> tuple[dict[str, Any] | None, str]:
    parsed = extract_json(text)
    parsed_dict = parsed if isinstance(parsed, dict) else {}
    code = parsed_dict.get("harness_code", parsed_dict.get("code"))
    if not code and isinstance(parsed_dict.get("files"), dict):
        code = parsed_dict["files"].get("harness.py")
    # Direct source is the primary protocol. Keep the JSON path for old
    # responses so archived/replayed proposals remain compatible.
    if not isinstance(code, str):
        direct_code = strip_code_fence(text)
        if "def run(" in direct_code:
            code = direct_code
            parsed_dict = {}
        else:
            return None, "agent response did not contain a complete harness source"
    if not isinstance(code, str):
        return None, "agent response did not contain harness_code"
    code = strip_code_fence(code)
    errors = []
    try:
        # Never execute top-level generated code in the orchestrator process.
        errors.extend(validate_harness_code(code))
        compile(code, 'candidate.py', 'exec')
    except (HarnessValidationError, ValueError, SyntaxError) as error:
        errors.append(str(error))
    if errors:
        return None, "; ".join(errors)
    if behavior_fingerprint(code) == behavior_fingerprint(parent_code):
        return None, "agent returned an unchanged parent"
    operation = str(parsed_dict.get("operation", expected_operation))
    if operation != expected_operation:
        return None, f"wrong operation {operation!r}; expected {expected_operation!r}"
    return {
        "code": code,
        "strategy_name": str(parsed_dict.get("strategy_name", f"code_only_{expected_operation}"))[:240],
        "rationale": str(parsed_dict.get("rationale", "direct executable code proposal"))[:1200],
        "failure_hypothesis": str(parsed_dict.get("failure_hypothesis", "see failure bank and source diff"))[:1200],
        "operation": operation,
    }, ""


def behavior_fingerprint(code):
    tree = ast.parse(code)
    # Ignore comments/formatting/docstrings, not arbitrary semantic equivalence.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def reference_literal_leak(code, rows):
    """Catch direct answer/input lookup literals.

    LongLaMP references are long enough for the old target-only check.  LaMP
    classification/rating targets are intentionally short, so checking the
    target vocabulary itself would reject legitimate category/rating logic.
    Instead, reject complete long training inputs and exact sample IDs as the
    lookup key.  This is a conservative direct-leak check, not a proof
    against arbitrary obfuscation or learned semantic memorization.
    """
    literals = [' '.join(n.value.split()) for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    literal_set = set(literals)
    for row in rows:
        gold = ' '.join(str(row.get('target', '')).split())
        generation_reference = row.get('benchmark') == 'longlamp' or row.get('task') in (4, 5)
        if (len(gold) >= 100 or (generation_reference and len(gold.split()) >= 4)) and any(gold in text for text in literals):
            return 'candidate embeds a complete training reference literal; infer a policy, not answers'
        sample_id = ' '.join(str(row.get('sample_id', '')).split())
        if len(sample_id) >= 8 and sample_id in literal_set:
            return 'candidate embeds a training sample_id lookup; infer a policy, not answer tables'
        current_input = ' '.join(str(row.get('input', '')).split())
        if len(current_input) >= 80 and any(current_input in text for text in literals):
            return 'candidate embeds a complete training input lookup; infer a policy, not answer tables'
    return None


def _fallback_hypotheses(operation: str, count: int, iteration: int = 1) -> list[str]:
    """Return concrete, black-box directions when the planner is unavailable.

    These are only a failure fallback for the Qwen3.8 planning request.  They
    are deliberately phrased as testable hypotheses rather than a fixed
    architecture, so an outage or malformed planning response does not send
    every branch through the old ``Independent branch`` no-guidance path.
    """
    macro = [
        "Test a query-conditioned hybrid memory policy that combines title/content lexical overlap with semantic similarity and diversity reranking; compare whether the current failures are caused by retrieving examples that share words but not task structure.",
        "Test a compact task-conditioned profile card that separates author style, recurring terminology, and content evidence instead of concatenating the same history into every prompt; verify changes on the worst traces.",
        "Test a black-box generation workflow change between direct generation and a short draft/revision or candidate-selection pass, using extra calls only when the failure bank shows missing structure or terminology.",
        "Test a within-user task router that selects retrieval/context and generation settings from the current title and profile evidence, with a direct-generation fallback for uncertain tasks; do not hardcode sample answers.",
        "Test a preference-guided selection policy that generates a small set of outputs and scores them with an explicit task-fidelity and user-style checklist, while preserving a safe fallback when judging is unreliable.",
        "Test a lightweight multi-view or graph-like memory representation of entities, topics, and relations from the profile, and compare it against the parent's simpler memory on the observed failures.",
    ]
    micro = [
        "Repair retrieval/context detail using the failure bank: remove irrelevant examples, cap long evidence, and preserve the task constraints and high-value terminology; measure whether regressions come from context dilution.",
        "Repair the profile reranking rule by balancing title overlap, semantic relevance, recency, and diversity rather than taking one signal alone; test the changed selection on the parent failures.",
        "Repair prompt construction so task instructions, user-style guidance, historical evidence, and output contract are clearly separated; keep historical content as soft evidence rather than answer text.",
        "Repair generation length and structure control using the observed output lengths and per-sample errors; preserve concrete task entities and add only a minimal black-box verification step if it addresses a measured failure.",
        "Repair empty-profile, irrelevant-profile, and failed-extra-call fallbacks so they return a valid task-focused answer without copying history or silently changing the task.",
        "Repair candidate selection or call budgeting using the failure bank: spend extra QA calls only on uncertain cases and keep the deterministic direct path when extra reasoning has not shown a benefit.",
    ]
    catalog = macro if operation == "macro_strategy" else micro
    offset = max(0, int(iteration) - 1) % len(catalog)
    return [catalog[(offset + index) % len(catalog)] for index in range(max(1, int(count)))]


def _parse_hypotheses(response: str, count: int) -> list[str]:
    """Parse the planner's comment-only response without executing it."""
    hypotheses = re.findall(
        r'^\s*#\s*HYPOTHESIS\s+\d+\s*:\s*(.+)$',
        response,
        flags=re.M | re.I,
    )
    if not hypotheses:
        # Backward-compatible literal form, parsed without executing anything.
        tree = ast.parse(strip_code_fence(response))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "HYPOTHESES"
                for target in node.targets
            ):
                hypotheses = ast.literal_eval(node.value)
                break
    if (
        not isinstance(hypotheses, list)
        or len(hypotheses) != count
        or any(not isinstance(hypothesis, str) or not hypothesis.strip()
               for hypothesis in hypotheses)
        or len(set(hypotheses)) != count
    ):
        raise ValueError("planner must supply distinct nonempty hypotheses")
    return [hypothesis.strip() for hypothesis in hypotheses]


def parse_operation_choice(response):
    """Parse a data-only Python decision, never execute agent output."""
    text = response.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:])
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3]
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Formatting preambles are not executable code. Recover only the two
        # literal assignment lines; malformed quotes still require a repair.
        lines = re.findall(r'^\s*(?:operation|reason)\s*=.*$', text, re.M)
        if len(lines) != 2:
            raise ValueError('Decision needs exactly two unambiguous literal assignments')
        tree = ast.parse('\n'.join(line.strip() for line in lines))
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError('Operation decision must contain only literal assignments')
        name = node.targets[0].id
        if name not in ('operation', 'reason') or name in values:
            raise ValueError('Unexpected or duplicate operation decision field')
        values[name] = ast.literal_eval(node.value)
    if values.get('operation') not in OPERATION_NAMES:
        raise ValueError('Choose exactly one supported operation')
    if not isinstance(values.get('reason'), str) or not values['reason'].strip():
        raise ValueError('Operation choice requires an evidence-grounded reason')
    return values


def choose_operation(*, parent_code, iteration, current_summary, failure_bank,
                     history, archive, task_context, agent_api_url, agent_api_model,
                     agent_timeout, request_gate, artifact_dir):
    """Select one operation using historical adaptation evidence only."""
    import uuid
    from contextlib import nullcontext
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    def record(value):
        (destination / (uuid.uuid4().hex + '.json')).write_text(
            json.dumps(value, ensure_ascii=False, indent=2))
    evidence = dict(iteration=iteration, task=task_context,
                    adaptation_summary=current_summary,
                    failure_bank=failure_bank.prompt_view(),
                    history=_history_summary(history), archive=_archive_summary(archive))
    prompt = '''Choose exactly ONE operation for this user's next evolution round.
macro_strategy: substantially change the personalization strategy or architecture.
micro_repair: retain the strategy and fix or refine its implementation.
Neither operation is preferred. There is no alternation schedule or quota; consecutive
rounds may choose the same operation. Ground the choice in the evidence, uncertainty,
and prior attempts. All supplied diagnostics come from historical-profile adaptation.
Never request or use the current held-out test answer or test performance.
Return only two Python literal assignments (no implementation yet):
operation = "<choose macro_strategy or micro_repair>"
reason = "evidence supporting the choice"
EVIDENCE:\n'''
    def scrub(value):
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items() if k != 'user_id'}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value
    evidence = scrub(evidence)
    # Match the context manager's section contract, preserving the full parent
    # and supporting progressive compression of history rather than failure.
    prompt += f"Step {iteration}; operation=selection; branch=0.\n"
    prompt += 'Task:\n' + task_context + '\n\nCurrent branch evaluation;\n'
    prompt += json.dumps(evidence['adaptation_summary'], ensure_ascii=False)
    prompt += '\n\nTraining feedback sampled ACROSS previous steps\n'
    prompt += json.dumps(evidence['failure_bank'], ensure_ascii=False)
    prompt += '\n\nEvolution history (adaptation only)\n' + json.dumps(evidence['history'], ensure_ascii=False)
    prompt += '\n\nHistorical alternatives with actual source/diffs\n' + json.dumps(evidence['archive'], ensure_ascii=False)
    prompt += '\n\nCurrent branch source (replaceable):\n```python\n' + parent_code + '\n```'
    with request_gate if request_gate is not None else nullcontext():
        prompt = fit_agent_context(prompt, agent_api_url, record, agent_api_model)
        response = agent_chat(agent_api_url, agent_api_model, prompt,
                              timeout=agent_timeout, retries=2, retry_wait=2,
                              max_tokens=None, record=record)
    for attempt in range(3):
        try:
            return parse_operation_choice(response)
        except (ValueError, SyntaxError) as error:
            record({'phase': 'operation_format_failed', 'attempt': attempt, 'error': str(error)})
            if attempt == 2:
                raise
            repair = ('Repair ONLY the serialization of the following operation decision. '
                      'Preserve its chosen operation and rationale. Output exactly two valid Python '
                      'string-literal assignments: operation and reason. Escape internal quotes, '
                      'do not prepend prose or change the decision.\n' + response)
            with request_gate if request_gate is not None else nullcontext():
                response = agent_chat(agent_api_url, agent_api_model, repair, timeout=agent_timeout,
                                      retries=2, retry_wait=2, max_tokens=None, record=record)


def plan_hypotheses(*, parent_code, operation, iteration, count, current_summary,
                    failure_bank, history, archive, strategy_library,
                    agent_api_url, agent_api_model, agent_timeout, agent_retries,
                    agent_retry_wait, agent_max_tokens, request_gate=None,
                    artifact_dir=None, runtime_model='Qwen2.5-7B-Instruct',
                    runtime_context=8192, task_context=None,
                    objective_description=None, evaluation_boundary=None) -> list[str]:
    """Plan the whole operation before splitting candidates across parents.

    Archive-beam often gives one proposal slot to each parent.  Planning inside
    ``propose_candidates(count=1)`` therefore used to be skipped and every
    branch received the same placeholder hypothesis.  This operation-level
    planner is called once for all slots, then the runner passes one concrete
    hypothesis to each parent-specific implementation request.
    """
    import uuid
    from contextlib import nullcontext

    def record(value):
        if artifact_dir is not None:
            destination = Path(artifact_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / (uuid.uuid4().hex + ".json")).write_text(
                json.dumps(value, ensure_ascii=False, indent=2)
            )

    def request(prompt):
        with request_gate if request_gate is not None else nullcontext():
            if artifact_dir is not None:
                prompt = fit_agent_context(prompt, agent_api_url, record, agent_api_model)
            return agent_chat(
                agent_api_url,
                agent_api_model,
                prompt,
                timeout=agent_timeout,
                retries=agent_retries,
                retry_wait=agent_retry_wait,
                max_tokens=agent_max_tokens,
                record=record,
            )

    count = max(1, int(count))
    fallback = _fallback_hypotheses(operation, count, iteration)
    base = evolution_prompt(
        parent_code=parent_code,
        operation=operation,
        iteration=iteration,
        branch="planning",
        current_summary=current_summary,
        failure_bank=failure_bank,
        history=history,
        archive=archive,
        strategy_library=strategy_library,
        agent_model=agent_api_model,
        runtime_model=runtime_model,
        runtime_context=runtime_context,
        task_context=task_context,
        objective_description=objective_description,
        evaluation_boundary=evaluation_boundary,
    )
    try:
        response = request(base + f"""
Before implementation, propose exactly {count} genuinely DISTINCT evidence-grounded hypotheses
for this operation. These hypotheses will be assigned to different parent branches, so each
must identify one causal change and a concrete test. Cover different directions when the
evidence permits; do not return generic placeholders or repeat a previous failure.
For this planning request ONLY return Python comment lines, one per hypothesis:
{chr(10).join(f'# HYPOTHESIS {i}: distinct hypothesis, supporting evidence, proposed test' for i in range(1, count + 1))}
Continue to the requested count. Do not quote strings or implement the harness yet.
""")
        return _parse_hypotheses(response, count)
    except Exception as error:
        # A planner failure should reduce sophistication, not reduce the
        # operation to four identical/no-guidance requests.
        record({
            "phase": "planning_failed",
            "error": str(error),
            "fallback_hypotheses": fallback,
        })
        return fallback


def propose_candidates(*, parent_code, operation, iteration, count, current_summary,
                       failure_bank, history, archive, strategy_library, agent_api_url,
                       agent_api_model, agent_timeout, agent_retries, agent_retry_wait,
                       agent_concurrency, agent_max_tokens, request_gate=None, artifact_dir=None,
                       planned_hypotheses=None, runtime_model='Qwen2.5-7B-Instruct',
                       runtime_context=8192, task_context=None,
                       objective_description=None, evaluation_boundary=None):
    import uuid
    from contextlib import nullcontext
    def record(value):
        if artifact_dir is not None:
            destination = Path(artifact_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / (uuid.uuid4().hex + ".json")).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    def request(prompt):
        with request_gate if request_gate is not None else nullcontext():
            if artifact_dir is not None:
                prompt = fit_agent_context(prompt, agent_api_url, record, agent_api_model)
            return agent_chat(agent_api_url, agent_api_model, prompt, timeout=agent_timeout,
                              retries=agent_retries, retry_wait=agent_retry_wait,
                              max_tokens=agent_max_tokens, record=record)

    base = evolution_prompt(parent_code=parent_code, operation=operation, iteration=iteration,
                            branch="planning", current_summary=current_summary,
                            failure_bank=failure_bank, history=history, archive=archive,
                            strategy_library=strategy_library, agent_model=agent_api_model,
                            runtime_model=runtime_model, runtime_context=runtime_context,
                            task_context=task_context,
                            objective_description=objective_description,
                            evaluation_boundary=evaluation_boundary)
    hypotheses = []
    planning_error = None
    if planned_hypotheses is not None:
        hypotheses = [str(hypothesis).strip() for hypothesis in planned_hypotheses]
        if len(hypotheses) != count or any(not hypothesis for hypothesis in hypotheses):
            raise ValueError("planned_hypotheses must contain one hypothesis per candidate")
    elif count > 1:
        try:
            response = request(base + f"""
Before implementation, propose exactly {count} genuinely DISTINCT evidence-grounded hypotheses.
Choose the directions yourself, not a predefined architecture list. Avoid repeating historical failures.
For this planning request ONLY return Python comment lines, one per hypothesis:
# HYPOTHESIS 1: your hypothesis, supporting evidence, and proposed test
# HYPOTHESIS 2: a different hypothesis, evidence, and proposed test
Continue to the requested count. Do not quote strings or implement the harness yet.
Each item should explain a different causal change and how it can be tested.
""")
            hypotheses = _parse_hypotheses(response, count)
        except Exception as error:
            planning_error = str(error)
            hypotheses = []
            record({"phase": "planning_failed", "error": planning_error})
    if not hypotheses:
        hypotheses = _fallback_hypotheses(operation, count, iteration)

    def one(branch):
        prompt = evolution_prompt(parent_code=parent_code, operation=operation, iteration=iteration,
                                  branch=branch, current_summary=current_summary, failure_bank=failure_bank,
                                  history=history, archive=archive, strategy_library=strategy_library,
                                  agent_model=agent_api_model, runtime_model=runtime_model,
                                  runtime_context=runtime_context, task_context=task_context,
                                  objective_description=objective_description,
                                  evaluation_boundary=evaluation_boundary)
        prompt += "\nYour hypothesis: " + hypotheses[branch]
        prompt += "\nOther branches (do not duplicate their central change): " + json.dumps(
            [h for i, h in enumerate(hypotheses) if i != branch], ensure_ascii=False)
        try:
            response = request(prompt)
        except Exception as error:
            return dict(branch=branch, valid=False, error=str(error), failure_kind="service_request",
                        hypothesis=hypotheses[branch])
        proposal, error = parse_proposal(response, operation, parent_code)
        repaired_response = None
        if proposal is None:
            try:
                repaired_response = request(
                    prompt + "\nRepair the rejected candidate below using the compiler feedback. Preserve its hypothesis.\n"
                    + "\nCompiler feedback: " + error + "\nSource:\n" + response
                    + "\nReturn only complete Python source; no minimum length.")
                proposal, error = parse_proposal(repaired_response, operation, parent_code)
            except Exception as repair_error:
                error = str(repair_error)
        if proposal is None:
            return dict(branch=branch, valid=False, error=error, raw_response=response,
                        repaired_response=repaired_response, hypothesis=hypotheses[branch])
        proposal.update(branch=branch, valid=True, hypothesis=hypotheses[branch], planning_error=planning_error)
        return proposal

    with ThreadPoolExecutor(max_workers=max(1, min(agent_concurrency, count))) as pool:
        result = list(pool.map(one, range(count)))
    seen = {behavior_fingerprint(parent_code)}
    for proposal in result:
        if not proposal.get("valid"):
            continue
        fingerprint = behavior_fingerprint(proposal["code"])
        if fingerprint in seen:
            proposal.update(valid=False, error="duplicate candidate AST (ignoring comments/docstrings)")
        seen.add(fingerprint)
    return result


def evaluate_code(
    code: str,
    rows: list[dict[str, Any]],
    qa_client: VLLMOpenAIClient,
    *,
    qa_max_calls: int,
    qa_concurrency: int,
    label: str,
    sample_timeout: float = 300,
) -> list[dict[str, Any]]:
    errors = validate_harness_code(code)
    if errors:
        raise HarnessValidationError('; '.join(errors))

    def one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, row = item
        qa = VLLMQA(qa_client, max_calls=qa_max_calls,
                    embedding_client=getattr(qa_client, 'embedding_client', None))
        result = {
            "user_id": str(row.get("user_id", "")),
            "sample_id": str(row.get("sample_id", index)),
            "input": row.get("input", ""),
            "profile": row.get("profile", []),
            "target": row.get("target", ""),
            "source_split": row.get("source_split", "unknown"),
            "prediction": "",
            "qa_calls": 0,
            "qa_trace": [],
            "error": "",
        }
        try:
            result["prediction"] = run_isolated(code, runtime_row(row), qa, timeout=sample_timeout)
            try:
                from .profile_protocol import task_hint_loss
            except ImportError:
                from profile_protocol import task_hint_loss
            lost = task_hint_loss(row, qa.trace)
            if lost:
                raise ValueError('Task information lost before LLM calls: ' + repr(lost))
        except Exception as error:  # candidate failures become failure-bank evidence
            result["error"] = f"{type(error).__name__}: {error}"[:500]
        result["qa_calls"] = qa.calls
        result["qa_trace"] = list(qa.trace)[:8]
        result['input_missing'] = bool(row.get('input_missing', False))
        return index, result

    started = time.perf_counter()
    workers = max(1, min(int(qa_concurrency), len(rows) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(one, enumerate(rows)))
    values.sort(key=lambda item: item[0])
    output = [value for _, value in values]
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "phase": "blackbox_target_eval",
        "label": label,
        "rows": len(output),
        "seconds": round(elapsed, 2),
        "rows_per_second": round(len(output) / max(1e-9, elapsed), 4),
        "qa_concurrency": workers,
    }, ensure_ascii=False), flush=True)
    return output


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
