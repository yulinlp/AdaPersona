"""Build a same-protocol LongLaMP leaderboard before evaluating generalization.

This runner deliberately evaluates every method with the same frozen target
model, rows, user profiles, decoding budget, and ROUGE implementation.  The
literature numbers for GLASS/CoSteer are useful references, but are not a
valid gate for an RSI run until their split/backbone/task protocol is
reproduced.  The local gate therefore compares RSI with the strongest
non-training harness in this exact run and records the literature caveat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from longlamp_rsi import (
        SEED_HARNESS,
        VLLMOpenAIClient,
        generate_batch,
        generate_records,
        load_jsonl,
        normalize_spec,
        recent,
        write_jsonl,
    )
except ModuleNotFoundError:  # import as ``scripts.benchmark_longlamp_nontraining``
    from scripts.longlamp_rsi import (
        SEED_HARNESS,
        VLLMOpenAIClient,
        generate_batch,
        generate_records,
        load_jsonl,
        normalize_spec,
        recent,
        write_jsonl,
    )

try:
    from evolution_metrics import (
        METRIC_PROTOCOL,
        OBJECTIVE_PROTOCOL,
        OBJECTIVE_WEIGHTS,
        row_metrics,
        weighted_metric_score,
    )
except ModuleNotFoundError:  # pragma: no cover - package-style invocation
    from scripts.evolution_metrics import (
        METRIC_PROTOCOL,
        OBJECTIVE_PROTOCOL,
        OBJECTIVE_WEIGHTS,
        row_metrics,
        weighted_metric_score,
    )


DEFAULT_METHODS = (
    "full_context",
    "rag_bm25_k2",
    "pag_bm25_k2",
    "cot",
)


def is_pag_method(method: str) -> bool:
    return method.startswith("pag_bm25_k") or method.startswith("pag_qwen38_bm25_k")


def pag_retrieval_k(method: str) -> int:
    for prefix in ("pag_bm25_k", "pag_qwen38_bm25_k"):
        if method.startswith(prefix):
            try:
                return int(method.removeprefix(prefix))
            except ValueError as error:
                raise ValueError(f"Invalid PAG method: {method}") from error
    raise ValueError(f"Not a PAG method: {method}")


def profile_key(profile: list[dict[str, Any]]) -> str:
    """Stable key for caching one leakage-safe profile view."""
    payload = json.dumps(profile or [], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_summary_prompt(profile: list[dict[str, Any]], max_items: int) -> str:
    """Build the inference-only profile summarizer prompt used by PAG."""
    selected = recent(list(profile or []), max_items)
    if not selected:
        return (
            "Summarize the user's writing preferences from the history below. "
            "There is no history, so return: No reliable user profile is available."
        )
    items = []
    for index, item in enumerate(selected, 1):
        title = " ".join(str(item.get("title", "")).split())
        abstract = " ".join(str(item.get("abstract", "")).split())
        if len(abstract) > 700:
            abstract = abstract[:700].rsplit(" ", 1)[0] + "..."
        items.append(f"[History {index}] Title: {title}\nAbstract: {abstract}")
    return (
        "You are an inference-time user-profile summarizer for academic abstract generation.\n"
        "Read the historical papers written by one author and produce a compact profile that "
        "another language model can use when writing a new abstract. Capture stable writing "
        "organization, technical density, rhetorical moves, typical level of detail, and "
        "recurring research interests. Do not answer any paper task. Do not copy a complete "
        "sentence or invent facts. Keep the summary under 180 words and return only the "
        "profile summary.\n\n"
        "Historical papers:\n" + "\n\n".join(items)
    )


def load_summary_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"PAG summary cache must be a JSON object: {path}")
    return {str(key): str(summary) for key, summary in value.items()}


def save_summary_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True))
    temp.replace(path)


def build_pag_summaries(
    rows: dict[str, list[dict[str, Any]]],
    model: Any,
    cache_path: Path,
    *,
    max_items: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> dict[str, str]:
    """Generate/cached PAG summaries using the frozen target model only."""
    cache = load_summary_cache(cache_path)
    profiles: dict[str, list[dict[str, Any]]] = {}
    for user_rows in rows.values():
        for row in user_rows:
            profile = list(row.get("profile") or [])
            profiles.setdefault(profile_key(profile), profile)
    missing = [key for key in sorted(profiles) if key not in cache]
    if missing:
        prompts = [profile_summary_prompt(profiles[key], max_items) for key in missing]
        summaries = generate_batch(
            None,
            model,
            prompts,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            system="You summarize a user's historical academic writing for personalization.",
        )
        if len(summaries) != len(missing):
            raise RuntimeError(f"PAG summary count mismatch: {len(summaries)} != {len(missing)}")
        cache.update({key: summary.strip() for key, summary in zip(missing, summaries)})
        save_summary_cache(cache_path, cache)
    return cache


def render_history_item(item: dict[str, Any], index: int, max_chars: int) -> str:
    title = " ".join(str(item.get("title", "")).split())
    abstract = " ".join(str(item.get("abstract", "")).split())
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"[History {index}] Title: {title}\nAbstract: {abstract}"


def full_context_prompt(
    row: dict[str, Any],
    *,
    max_context_chars: int,
    max_item_chars: int,
) -> str:
    """Render as much of the available history as the frozen 8K context allows."""
    profile = list(row.get("profile") or [])
    selected = []
    used = 0
    for index, item in enumerate(recent(profile, len(profile)), 1):
        block = render_history_item(item, index, max_item_chars)
        extra = len(block) + (2 if selected else 0)
        if selected and used + extra > max_context_chars:
            break
        if not selected and extra > max_context_chars:
            block = block[:max_context_chars]
        selected.append(block)
        used += len(block) + (2 if len(selected) > 1 else 0)
    history = "\n\n".join(selected) or "No historical writing is available."
    return (
        "You are an expert academic writer.\n"
        "Write a complete, publication-quality abstract for the current task.\n"
        "Return only the final abstract, with no analysis, plan, or preface.\n"
        "Use the historical writing as soft evidence for the author's style and "
        "organization. Preserve the concrete entities in the current task and do "
        "not copy unrelated historical claims.\n\n"
        "Full available user history:\n" + history
        + "\n\nCurrent task:\n" + str(row.get("input", ""))
    )


def cot_plan_prompt(
    row: dict[str, Any],
    *,
    max_context_chars: int,
    max_item_chars: int,
) -> str:
    base = full_context_prompt(
        row,
        max_context_chars=max_context_chars,
        max_item_chars=max_item_chars,
    )
    return (
        base.replace(
            "Return only the final abstract, with no analysis, plan, or preface.",
            "Do not write the final abstract yet. Produce a concise internal writing "
            "plan for a second writer. Identify the task entities, a safe abstract "
            "organization, and the author's style cues. Do not invent experimental "
            "results or copy historical sentences. Return only the plan.",
        )
        + "\n\nPlanning pass:"
    )


def load_model(model_name: str, manifest_path: Path, transformers_path_override: str = ""):
    import torch

    manifest = {item["name"]: item for item in json.loads(manifest_path.read_text())}
    if model_name not in manifest:
        raise KeyError(f"Unknown model {model_name}")
    item = manifest[model_name]
    # The manifest records the newer Transformers tree used by the original
    # Qwen3 pilot.  For this benchmark, prefer the environment's compatible
    # Transformers unless the caller explicitly overrides it; otherwise a
    # stale safetensors dependency in that private tree can prevent even
    # Qwen2.5 checkpoints from loading.
    transformers_path = transformers_path_override
    if transformers_path and str(transformers_path) not in sys.path:
        sys.path.insert(0, str(transformers_path))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(item["local_path"]), use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"device_map": "auto"}
    try:
        # Newer Transformers renamed this argument to ``dtype``.
        model = AutoModelForCausalLM.from_pretrained(
            str(item["local_path"]), dtype=torch.bfloat16, **model_kwargs
        )
    except TypeError as error:
        if "unexpected keyword argument 'dtype'" not in str(error):
            raise
        # Qwen2.5 is also supported by the older, CUDA-12-compatible
        # environment used on gpu01.
        model = AutoModelForCausalLM.from_pretrained(
            str(item["local_path"]), torch_dtype=torch.bfloat16, **model_kwargs
        )
    model.eval()
    return tokenizer, model


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["user_id"])].append(row)
    for user in result:
        result[user].sort(key=lambda row: str(row.get("sample_id", "")))
    return dict(result)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row_metrics(row) for row in rows]
    users: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row, metric in zip(rows, metrics):
        users[str(row["user_id"])].append(metric)
    if not metrics:
        return {"n": 0, "users": 0}
    micro = {
        key: sum(item[key] for item in metrics) / len(metrics)
        for key in metrics[0]
    }
    micro["weighted_score"] = sum(weighted_metric_score(item) for item in metrics) / len(metrics)
    macro = {
        key: sum(sum(item[key] for item in values) / len(values) for values in users.values())
        / len(users)
        for key in metrics[0]
    }
    macro["weighted_score"] = sum(
        sum(weighted_metric_score(item) for item in values) / len(values)
        for values in users.values()
    ) / len(users)
    return {
        "n": len(metrics),
        "users": len(users),
        "micro": {key: round(value, 6) for key, value in micro.items()},
        "macro_user": {key: round(value, 6) for key, value in macro.items()},
    }


def direct_spec(
    *,
    evidence_policy: str,
    evidence_k: int,
    abstraction: str = "none",
    stage: str = "direct",
) -> dict[str, Any]:
    spec = normalize_spec(SEED_HARNESS)
    if spec is None:
        raise ValueError("SEED_HARNESS is invalid")
    spec["evidence"].update({
        "policy": evidence_policy,
        "k": evidence_k,
        "query_mode": "title",
    })
    spec["abstraction"]["mode"] = abstraction
    spec["interface"]["layout"] = "examples_only"
    spec["router"] = {"mode": "static", "threshold": 0.25, "fallback": "none"}
    spec["reasoning"]["stages"] = [stage]
    spec["output"] = {"contract": "abstract_only", "length": "free"}
    spec["graph"]["mode"] = "linear"
    spec["budget"].update({
        "max_calls": {"direct": 1, "draft_revision": 2}[stage],
        "max_input_tokens": 4096,
        "max_new_tokens": 256,
    })
    spec["instruction"] = (
        "Write the abstract for the current task. Preserve concrete task entities "
        "and do not copy unsupported claims from examples."
    )
    normalized = normalize_spec(spec)
    if normalized is None:
        raise ValueError("Could not normalize baseline spec")
    return normalized


def method_specs(method: str, users: list[str], rsi_specs: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if method == "non_personalized":
        spec = direct_spec(evidence_policy="none", evidence_k=0)
        return {user: spec for user in users}
    if method.startswith("rag_bm25_k"):
        try:
            k = int(method.removeprefix("rag_bm25_k"))
        except ValueError as error:
            raise ValueError(f"Invalid RAG method: {method}") from error
        spec = direct_spec(evidence_policy="bm25", evidence_k=k)
        return {user: spec for user in users}
    if method.startswith("pag_bm25_k") or method.startswith("pag_qwen38_bm25_k"):
        k = pag_retrieval_k(method)
        spec = direct_spec(evidence_policy="bm25", evidence_k=k)
        # The row-specific LLM-generated summary is attached when records are
        # materialized below.  It is not a target or an evolution signal.
        spec["interface"]["layout"] = "examples_only"
        return {user: spec for user in users}
    if method in {"full_context", "cot", "cot_qwen38"}:
        # These two baselines use dedicated prompts below rather than the
        # compact evidence renderer in ``generate_records``.  Keep a normal
        # harness spec in the artifact so the output remains auditable, but
        # do not let the generic RAG path silently redefine the methods.
        spec = direct_spec(evidence_policy="none", evidence_k=0)
        spec["module_guidance"] = {
            "full_context": "Inference-only full available user history prompt.",
            "cot": "Inference-only plan-then-write prompt using full available user history.",
        }[method]
        return {user: spec for user in users}
    if method == "full_history_recent8":
        # The frozen target service has an 8K context.  This is a practical,
        # reproducible recent-history approximation, not the 32K literature
        # setting called Full History.
        spec = direct_spec(evidence_policy="recent", evidence_k=8)
        spec["interface"]["max_example_chars"] = 800
        return {user: spec for user in users}
    if method == "seed_draft_revision":
        spec = direct_spec(evidence_policy="bm25", evidence_k=3, abstraction="style_card", stage="draft_revision")
        spec["interface"]["layout"] = "style_then_examples"
        spec["output"]["length"] = "profile_adaptive"
        spec["instruction"] = "Match the author's content organization and writing style while answering the current task."
        return {user: spec for user in users}
    if method == "rsi":
        if rsi_specs is None:
            raise ValueError("--specs is required when evaluating rsi")
        result: dict[str, dict[str, Any]] = {}
        for user in users:
            spec = normalize_spec(rsi_specs.get(user))
            if spec is None:
                raise ValueError(f"No valid RSI spec for user {user}")
            result[user] = spec
        return result
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen25_7b")
    parser.add_argument(
        "--target-backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="Target generation backend. vllm uses a local OpenAI-compatible server.",
    )
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="", help="Served model name; defaults to manifest model_id.")
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    parser.add_argument("--vllm-concurrency", type=int, default=16)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("data/model_manifest.json"),
    )
    parser.add_argument(
        "--transformers-path",
        default="",
        help="Optional Transformers checkout to prepend to PYTHONPATH.",
    )
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--split", default="evolve_train")
    parser.add_argument("--specs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument(
        "--users-from",
        type=Path,
        default=None,
        help="JSON dict/list of user ids to evaluate, useful for a matched pilot.",
    )
    parser.add_argument("--rows-per-user", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pag-summary-items", type=int, default=16)
    parser.add_argument("--pag-summary-max-input-tokens", type=int, default=4096)
    parser.add_argument("--pag-summary-max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--prompt-vllm-base-url",
        default="http://gpu02:18012/v1",
        help="Local prompt-construction model endpoint for *_qwen38_front methods.",
    )
    parser.add_argument("--prompt-vllm-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--prompt-vllm-timeout", type=float, default=1200.0)
    parser.add_argument("--prompt-vllm-concurrency", type=int, default=2)
    parser.add_argument(
        "--full-context-max-chars",
        type=int,
        default=22000,
        help="Approximate prompt budget for full_context/cot under the frozen 8K target context.",
    )
    parser.add_argument("--full-context-item-max-chars", type=int, default=700)
    parser.add_argument("--cot-plan-max-new-tokens", type=int, default=192)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--require-rsi-over-best",
        action="store_true",
        help="Exit nonzero unless RSI beats the best same-protocol non-training method.",
    )
    parser.add_argument("--min-margin", type=float, default=0.0)
    args = parser.parse_args()

    if "rsi" in args.methods and args.specs is None:
        raise ValueError("--specs is required when methods include rsi")
    rsi_specs = json.loads(args.specs.read_text()) if args.specs else None
    rows = grouped(load_jsonl(args.pilot_dir / f"{args.split}.jsonl"))
    users = sorted(rows)
    if args.users_from is not None:
        selected_payload = json.loads(args.users_from.read_text())
        if isinstance(selected_payload, dict):
            selected_users = {str(user) for user in selected_payload}
        elif isinstance(selected_payload, list):
            selected_users = {str(user) for user in selected_payload}
        else:
            raise ValueError("--users-from must contain a JSON object or list")
        users = [user for user in users if user in selected_users]
    if args.max_users > 0:
        users = users[: args.max_users]
    rows = {
        user: rows[user][: args.rows_per_user] if args.rows_per_user > 0 else rows[user]
        for user in users
    }
    if not users or not any(rows.values()):
        raise ValueError(f"No rows found for split {args.split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.target_backend == "vllm":
        # A local OpenAI-compatible service already identifies its served
        # model.  Do not require the unrelated historical Transformers
        # manifest for the V5 black-box baseline path.
        served_model = args.vllm_model or {
            "qwen25_7b": "Qwen2.5-7B-Instruct",
        }.get(args.model, args.model)
        tokenizer = None
        model = VLLMOpenAIClient(
            args.vllm_base_url,
            served_model,
            timeout=args.vllm_timeout,
            concurrency=args.vllm_concurrency,
            chat_template_kwargs=(
                {"enable_thinking": False}
                if "qwen3" in served_model.lower()
                else None
            ),
        )
    else:
        tokenizer, model = load_model(args.model, args.model_manifest, args.transformers_path)

    prompt_model = model
    prompt_model_name = args.model
    qwen38_front_methods = {
        method for method in args.methods
        if method == "cot_qwen38" or method.startswith("pag_qwen38_bm25_k")
    }
    if qwen38_front_methods:
        if args.target_backend != "vllm":
            raise ValueError("Qwen3.8 front methods require --target-backend vllm")
        prompt_model = VLLMOpenAIClient(
            args.prompt_vllm_base_url,
            args.prompt_vllm_model,
            timeout=args.prompt_vllm_timeout,
            concurrency=args.prompt_vllm_concurrency,
            chat_template_kwargs=(
                {"enable_thinking": False}
                if "qwen3" in args.prompt_vllm_model.lower()
                else None
            ),
        )
        prompt_model_name = args.prompt_vllm_model

    pag_summaries: dict[str, str] = {}
    pag_qwen38_summaries: dict[str, str] = {}
    if any(method.startswith("pag_bm25_k") for method in args.methods):
        if not isinstance(model, VLLMOpenAIClient):
            raise ValueError("PAG summary generation currently requires the vLLM target backend")
        pag_summaries = build_pag_summaries(
            rows,
            model,
            args.output_dir / "pag_profile_summaries.json",
            max_items=max(1, args.pag_summary_items),
            max_input_tokens=max(2048, args.pag_summary_max_input_tokens),
            max_new_tokens=max(64, args.pag_summary_max_new_tokens),
        )
    if any(method.startswith("pag_qwen38_bm25_k") for method in args.methods):
        if not isinstance(prompt_model, VLLMOpenAIClient):
            raise ValueError("Qwen3.8 PAG summary generation requires the vLLM prompt backend")
        pag_qwen38_summaries = build_pag_summaries(
            rows,
            prompt_model,
            args.output_dir / "pag_profile_summaries_qwen38.json",
            max_items=max(1, args.pag_summary_items),
            max_input_tokens=max(2048, args.pag_summary_max_input_tokens),
            max_new_tokens=max(64, args.pag_summary_max_new_tokens),
        )

    standard_records: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    custom_records: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    metadata: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
    specs_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for method in args.methods:
        specs = method_specs(method, users, rsi_specs)
        specs_by_method[method] = specs
        for user in users:
            for row in rows[user]:
                index = len(metadata)
                row_spec = deepcopy(specs[user])
                if method.startswith("pag_bm25_k"):
                    row_spec["module_guidance"] = (
                        "PAG user profile summary (inference-only historical evidence):\n"
                        + pag_summaries[profile_key(list(row.get("profile") or []))]
                    )
                elif method.startswith("pag_qwen38_bm25_k"):
                    row_spec["module_guidance"] = (
                        "PAG user profile summary generated by Qwen3.8-27B "
                        "(inference-only historical evidence):\n"
                        + pag_qwen38_summaries[profile_key(list(row.get("profile") or []))]
                    )
                if method in {"full_context", "cot", "cot_qwen38"}:
                    custom_records[method].append((index, row, row_spec))
                else:
                    standard_records.append((index, row, row_spec))
                metadata.append((method, user, row, row_spec))

    predictions: dict[int, str] = {}
    if standard_records:
        predictions.update(generate_records(tokenizer, model, standard_records, batch_size=args.batch_size))

    # Full Context and CoT are explicit inference-only baselines.  They use
    # the same frozen target model and user history as RAG/PAG, but have their
    # own prompt construction so the comparison reflects the named methods.
    for method, records in custom_records.items():
        base_prompts = [
            full_context_prompt(
                row,
                max_context_chars=max(2000, args.full_context_max_chars),
                max_item_chars=max(200, args.full_context_item_max_chars),
            )
            for _, row, _ in records
        ]
        if method == "full_context":
            outputs = generate_batch(
                tokenizer,
                model,
                base_prompts,
                max_input_tokens=7000,
                max_new_tokens=256,
                system="You are an expert academic writer.",
            )
        else:
            plan_model = prompt_model if method == "cot_qwen38" else model
            plans = generate_batch(
                None if isinstance(plan_model, VLLMOpenAIClient) else tokenizer,
                plan_model,
                [cot_plan_prompt(
                    row,
                    max_context_chars=max(2000, args.full_context_max_chars),
                    max_item_chars=max(200, args.full_context_item_max_chars),
                ) for _, row, _ in records],
                max_input_tokens=7000,
                max_new_tokens=max(64, args.cot_plan_max_new_tokens),
                system="You are a careful academic writing planner. Return only a concise plan.",
            )
            final_prompts = [
                prompt
                + "\n\nInternal writing plan (follow it but do not mention it):\n"
                + plan
                + "\n\nNow write only the final abstract for the current task."
                for prompt, plan in zip(base_prompts, plans)
            ]
            outputs = generate_batch(
                tokenizer,
                model,
                final_prompts,
                max_input_tokens=7000,
                max_new_tokens=256,
                system="You are an expert academic writer.",
            )
        if len(outputs) != len(records):
            raise RuntimeError(f"{method} prediction count mismatch: {len(outputs)} != {len(records)}")
        predictions.update({index: output for (index, _, _), output in zip(records, outputs)})
    output_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, (method, user, row, spec) in enumerate(metadata):
        output_rows[method].append({
            "method": method,
            "user_id": user,
            "sample_id": row["sample_id"],
            "input": row.get("input", ""),
            "source_split": row.get("source_split", args.split),
            "target": row["target"],
            "prediction": predictions.get(index, ""),
            "harness": spec,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard: dict[str, Any] = {}
    for method, method_rows in output_rows.items():
        write_jsonl(args.output_dir / f"{method}.jsonl", method_rows)
        leaderboard[method] = summarize(method_rows)

    nontraining = {
        method: summary["macro_user"]["rougeL"]
        for method, summary in leaderboard.items()
        if method != "rsi"
    }
    best_method = max(nontraining, key=nontraining.get) if nontraining else None
    rsi_score = leaderboard.get("rsi", {}).get("macro_user", {}).get("rougeL")
    best_score = nontraining.get(best_method) if best_method else None
    margin = (rsi_score - best_score) if rsi_score is not None and best_score is not None else None
    report = {
        "model": args.model,
        "target_backend": args.target_backend,
        "vllm_base_url": args.vllm_base_url if args.target_backend == "vllm" else None,
        "prompt_model": prompt_model_name if qwen38_front_methods else None,
        "prompt_vllm_base_url": args.prompt_vllm_base_url if qwen38_front_methods else None,
        "split": args.split,
        "users": len(users),
        "rows": sum(len(values) for values in rows.values()),
        "methods": list(args.methods),
        "metric_protocol": METRIC_PROTOCOL,
        "objective_protocol": OBJECTIVE_PROTOCOL,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "pag_protocol": {
            "summary_model": args.model,
            "summary_items": args.pag_summary_items,
            "retriever": "BM25",
            "retrieval_k": 2,
            "summary_uses_current_target": False,
        },
        "pag_qwen38_protocol": {
            "summary_model": args.prompt_vllm_model,
            "summary_items": args.pag_summary_items,
            "retriever": "BM25",
            "retrieval_k": 2,
            "summary_uses_current_target": False,
            "final_generator": args.vllm_model or args.model,
        },
        "full_context_protocol": {
            "max_context_chars": args.full_context_max_chars,
            "max_item_chars": args.full_context_item_max_chars,
            "target_context_contract": "8K; prompt is char-bounded to leave generation room",
            "uses_current_target": False,
        },
        "cot_protocol": {
            "kind": "inference_only_plan_then_write",
            "plan_model": args.model,
            "plan_max_new_tokens": args.cot_plan_max_new_tokens,
            "uses_current_target": False,
            "parameter_updates": False,
        },
        "cot_qwen38_protocol": {
            "kind": "inference_only_plan_then_write",
            "plan_model": args.prompt_vllm_model,
            "plan_max_new_tokens": args.cot_plan_max_new_tokens,
            "final_generator": args.vllm_model or args.model,
            "uses_current_target": False,
            "parameter_updates": False,
        },
        "leaderboard": leaderboard,
        "sota_gate": {
            "definition": "best non-training method in this exact frozen-model/protocol run",
            "best_nontraining_method": best_method,
            "best_nontraining_macro_user_rougeL": best_score,
            "rsi_macro_user_rougeL": rsi_score,
            "rsi_margin": margin,
            "min_margin": args.min_margin,
            "passed": margin is not None and margin > args.min_margin,
        },
        "literature_warning": (
            "GLASS/CoSteer headline scores are not used as a numeric gate here because "
            "their backbone, task subset, split, and/or two-model architecture differ. "
            "Reproducing one of them under this protocol is required for a literal "
            "literature-SOTA claim."
        ),
    }
    (args.output_dir / "leaderboard.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(json.dumps(report["sota_gate"], ensure_ascii=False), flush=True)
    if args.require_rsi_over_best and not report["sota_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
