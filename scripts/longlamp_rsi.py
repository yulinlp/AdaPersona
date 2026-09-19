"""Utilities for recursive, user-specific LongLaMP harness evolution.

The harness is a small validated JSON object.  It is intentionally textual and
editable so an LLM evolver can propose workflow mutations without changing the
base model.  This module contains no training code; it only renders a harness,
runs batched generations, and computes lightweight lexical diagnostics.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
TITLE_RE = re.compile(r'title\s+["“](.*?)["”]', re.I | re.S)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "using",
    "via", "with", "we", "our", "their", "study", "new", "based", "paper",
}


SEED_HARNESS: dict[str, Any] = {
    # ``full`` preserves the original pilot's BM25 query.  The v3 seed uses
    # ``title`` so prompt-wrapper words do not compete with technical terms.
    "evidence": {"policy": "bm25", "k": 3, "query_mode": "full"},
    "abstraction": {"mode": "style_card"},
    "interface": {"layout": "style_then_examples", "max_example_chars": 1200},
    # ``static`` preserves the behavior of the original pilot.  The v3
    # evolver can mutate this into a topic-adaptive router without changing
    # the rest of the harness representation.
    "router": {"mode": "static", "threshold": 0.25, "fallback": "none"},
    "reasoning": {"stages": ["direct"]},
    "output": {"contract": "abstract_only", "length": "profile_adaptive"},
    "graph": {"mode": "linear"},
    "budget": {"max_calls": 1, "max_input_tokens": 4096, "max_new_tokens": 160},
    "instruction": "Match the author's content organization and writing style while answering the current task.",
    # Modular RSI may attach versioned module guidance here.  The field is
    # deliberately outside the fixed E/A/I/R/O;G interfaces: it is an
    # implementation artifact rendered by the selected module versions.
    "module_guidance": "",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@lru_cache(maxsize=200_000)
def tokens(text: str) -> tuple[str, ...]:
    """Tokenize once per repeated title/abstract string.

    LongLaMP rows reuse the same profile papers across many target rows.  The
    cache keeps BM25/style-card construction from re-running the regex over
    those papers for every harness evaluation.
    """
    return tuple(TOKEN_RE.findall(str(text).lower()))


def rouge_n(pred: str, ref: str, n: int) -> float:
    p = Counter(tuple(tokens(pred)[i : i + n]) for i in range(max(0, len(tokens(pred)) - n + 1)))
    r = Counter(tuple(tokens(ref)[i : i + n]) for i in range(max(0, len(tokens(ref)) - n + 1)))
    if not p or not r:
        return 0.0
    overlap = sum(min(value, r[key]) for key, value in p.items())
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(r.values()))
    return 2 * precision * recall / max(1e-12, precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    prev = [0] * (len(r) + 1)
    for item in p:
        cur = [0]
        for j, other in enumerate(r, 1):
            cur.append(prev[j - 1] + 1 if item == other else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / max(1e-12, precision + recall)


def bleu(pred: str, ref: str, max_n: int = 4) -> float:
    """Smoothed sentence BLEU-4 using the project's tokenizer.

    The pilot has no external metric dependency, so this is a compact
    reproducible implementation of clipped n-gram precision with brevity
    penalty.  Zero precisions use a small sentence-level smoothing value.
    """
    prediction, reference = tokens(pred), tokens(ref)
    if not prediction or not reference:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        predicted = Counter(tuple(prediction[index : index + n]) for index in range(len(prediction) - n + 1))
        target = Counter(tuple(reference[index : index + n]) for index in range(len(reference) - n + 1))
        total = sum(predicted.values())
        if total <= 0:
            continue
        clipped = sum(min(count, target[gram]) for gram, count in predicted.items())
        precisions.append(clipped / total if clipped else 1.0 / (2.0 * total))
    if not precisions:
        return 0.0
    brevity_penalty = 1.0 if len(prediction) > len(reference) else math.exp(1.0 - len(reference) / len(prediction))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / len(precisions))


def meteor_exact(pred: str, ref: str) -> float:
    """Dependency-free exact-token METEOR approximation.

    It uses METEOR's precision/recall harmonic mean and fragmentation penalty,
    but intentionally omits stemming and WordNet synonym matching.  The
    output is named ``meteor`` in reports and is deterministic across nodes.
    """
    prediction, reference = tokens(pred), tokens(ref)
    if not prediction or not reference:
        return 0.0
    reference_positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(reference):
        reference_positions[token].append(index)
    used: set[int] = set()
    aligned: list[tuple[int, int]] = []
    for prediction_index, token in enumerate(prediction):
        available = [index for index in reference_positions.get(token, []) if index not in used]
        if available:
            reference_index = available[0]
            used.add(reference_index)
            aligned.append((prediction_index, reference_index))
    matches = len(aligned)
    if not matches:
        return 0.0
    precision = matches / len(prediction)
    recall = matches / len(reference)
    harmonic = (10.0 * precision * recall) / max(1e-12, recall + 9.0 * precision)
    chunks = 1
    for previous, current in zip(aligned, aligned[1:]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return harmonic * max(0.0, 1.0 - penalty)


def row_metrics(row: dict[str, Any]) -> dict[str, float]:
    prediction, target = str(row.get("prediction", "")), str(row.get("target", ""))
    return {
        "rouge1": rouge_n(prediction, target, 1),
        "rouge2": rouge_n(prediction, target, 2),
        "rougeL": rouge_l(prediction, target),
        "bleu": bleu(prediction, target),
        "meteor": meteor_exact(prediction, target),
        "prediction_tokens": float(len(tokens(prediction))),
        "target_tokens": float(len(tokens(target))),
    }


def task_title(text: str) -> str:
    match = TITLE_RE.search(str(text))
    return " ".join(match.group(1).split()) if match else " ".join(str(text).split())[:240]


def retrieval_query(text: str) -> str:
    """Return the task content used for retrieval, excluding prompt scaffolding.

    LongLaMP inputs contain a natural-language wrapper (``Generate an
    abstract ... using the following items``).  Feeding that wrapper to BM25
    makes generic words compete with the title's technical terms.  The title
    is the only stable, leakage-free query shared by the real and synthetic
    tasks, so use it whenever it is present.
    """
    title = task_title(text)
    return title if title else str(text)


def title_coverage(input_text: str, prediction: str) -> float:
    """Fraction of informative title terms preserved in the generated text."""
    terms = [term for term in tokens(task_title(input_text)) if len(term) >= 4 and term not in STOPWORDS]
    if not terms:
        return 1.0
    output_terms = set(tokens(prediction))
    return sum(term in output_terms for term in set(terms)) / len(set(terms))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_spec(candidate: Any) -> dict[str, Any] | None:
    """Validate and normalize an evolver proposal.

    This validator is deliberately restrictive: arbitrary code or tool calls
    are outside this pilot.  It lets the evolver change the workflow while
    keeping every generated harness executable and comparable.
    """
    if isinstance(candidate, dict) and isinstance(candidate.get("spec"), dict):
        candidate = candidate["spec"]
    if not isinstance(candidate, dict):
        return None
    spec = deepcopy(SEED_HARNESS)
    for key in ("evidence", "abstraction", "interface", "router", "reasoning", "output", "graph", "budget"):
        if isinstance(candidate.get(key), dict):
            spec[key].update(candidate[key])
    if isinstance(candidate.get("instruction"), str):
        spec["instruction"] = " ".join(candidate["instruction"].split())[:600]
    if isinstance(candidate.get("module_guidance"), str):
        # Keep module internals richer than the legacy one-line instruction,
        # while still bounding the prompt contribution for reproducible runs.
        spec["module_guidance"] = candidate["module_guidance"].strip()[:6000]

    evidence = spec["evidence"]
    if evidence.get("policy") not in {"none", "recent", "bm25", "hybrid"}:
        return None
    evidence["k"] = max(0, min(8, _as_int(evidence.get("k"), 3)))
    if evidence.get("query_mode") not in {"full", "title"}:
        return None
    abstraction = spec["abstraction"]
    if abstraction.get("mode") not in {"none", "style_card", "content_card", "style_content"}:
        return None
    interface = spec["interface"]
    if interface.get("layout") not in {"examples_only", "style_then_examples", "examples_then_style", "cards_only"}:
        return None
    interface["max_example_chars"] = max(400, min(1800, _as_int(interface.get("max_example_chars"), 1200)))
    router = spec["router"]
    if router.get("mode") not in {"static", "topic_adaptive"}:
        return None
    try:
        threshold = float(router.get("threshold", 0.25))
    except (TypeError, ValueError):
        threshold = 0.25
    router["threshold"] = max(0.0, min(1.0, threshold))
    if router.get("fallback") not in {"none", "recent", "bm25", "hybrid"}:
        return None
    reasoning = spec["reasoning"]
    stages = reasoning.get("stages")
    if not isinstance(stages, list) or not stages or len(stages) > 3:
        return None
    allowed_stages = {"direct", "draft_revision", "draft_critique_revision"}
    if any(stage not in allowed_stages for stage in stages):
        return None
    # A single workflow is easier to attribute in this first evolution pilot.
    stages = [str(stages[-1])]
    reasoning["stages"] = stages
    output = spec["output"]
    if output.get("contract") not in {"abstract_only", "structured_abstract"}:
        return None
    if output.get("length") not in {"free", "profile_adaptive", "concise"}:
        return None
    graph = spec["graph"]
    if graph.get("mode") not in {"linear", "fallback"}:
        return None
    budget = spec["budget"]
    budget["max_calls"] = max(1, min(3, _as_int(budget.get("max_calls"), 1)))
    budget["max_input_tokens"] = max(2048, min(6144, _as_int(budget.get("max_input_tokens"), 4096)))
    budget["max_new_tokens"] = max(64, min(256, _as_int(budget.get("max_new_tokens"), 160)))
    required_calls = {"direct": 1, "draft_revision": 2, "draft_critique_revision": 3}[stages[0]]
    if budget["max_calls"] < required_calls:
        budget["max_calls"] = required_calls
    # Keep the prompt-level object small enough to inspect and log.
    spec["instruction"] = " ".join(str(spec.get("instruction", "")).split())[:600]
    return spec


def spec_json(spec: dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True)


def _profile_tokens(profile: list[dict[str, Any]]) -> list[str]:
    return tokens(" ".join(f"{item.get('title', '')} {item.get('abstract', '')}" for item in profile))


def bm25_top(profile: list[dict[str, Any]], query: str, k: int) -> list[dict[str, Any]]:
    if not profile or k <= 0:
        return []
    query_terms = Counter(tokens(query))
    docs = [tokens(f"{item.get('title', '')} {item.get('abstract', '')}") for item in profile]
    document_frequency = Counter(term for doc in docs for term in set(doc))
    average_length = sum(len(doc) for doc in docs) / max(1, len(docs))
    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for index, doc in enumerate(docs):
        term_frequency = Counter(doc)
        score = 0.0
        for term, query_frequency in query_terms.items():
            if not term_frequency[term]:
                continue
            idf = math.log(1.0 + (len(docs) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = term_frequency[term] + 0.9 * (0.25 + 0.75 * len(doc) / max(1.0, average_length))
            score += idf * term_frequency[term] * 1.9 / denominator * query_frequency
        scored.append((score, index, str(profile[index].get("id", "")), profile[index]))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[-1] for item in scored[:k]]


def recent(profile: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(profile, key=lambda item: (int(item.get("year", 0)), str(item.get("id", ""))), reverse=True)[:k]


def topic_confidence(profile: list[dict[str, Any]], query: str) -> float:
    """Estimate whether the profile contains topic-relevant evidence.

    This is deliberately lexical and reference-free.  It is not a quality
    score; it only tells a router whether the best historical record shares a
    useful fraction of the current title's informative terms.
    """
    query_terms = {
        term for term in tokens(retrieval_query(query))
        if len(term) >= 4 and term not in STOPWORDS
    }
    if not query_terms or not profile:
        return 0.0
    best = 0.0
    for item in profile:
        doc_terms = set(tokens(f"{item.get('title', '')} {item.get('abstract', '')}"))
        best = max(best, len(query_terms & doc_terms) / len(query_terms))
    return best


def select_evidence(profile: list[dict[str, Any]], query: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    policy = spec["evidence"]["policy"]
    router = spec.get("router", {})
    if router.get("mode") == "topic_adaptive" and topic_confidence(profile, query) < float(router.get("threshold", 0.25)):
        policy = router.get("fallback", "none")
    retrieval_input = query if spec["evidence"].get("query_mode") == "full" else retrieval_query(query)
    k = int(spec["evidence"]["k"])
    if policy == "none":
        return []
    if policy == "recent":
        return recent(profile, k)
    if policy == "bm25":
        return bm25_top(profile, retrieval_input, k)
    # Hybrid evidence keeps a recent anchor and fills the remaining slots with
    # query-relevant examples.  Duplicate records are removed by id.
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in recent(profile, max(1, k // 2)) + bm25_top(profile, retrieval_input, k):
        item_id = str(item.get("id", ""))
        if item_id in seen:
            continue
        seen.add(item_id)
        chosen.append(item)
        if len(chosen) >= k:
            break
    return chosen


def style_card(profile: list[dict[str, Any]]) -> str:
    abstracts = [str(item.get("abstract", "")).strip() for item in profile if item.get("abstract")]
    if not abstracts:
        return "No writing-style statistics are available."
    words = [len(tokens(text)) for text in abstracts]
    sentences = [max(1, len(re.findall(r"[.!?]+", text))) for text in abstracts]
    first_person = sum(bool(re.search(r"\b(I|we|our|my)\b", text, re.I)) for text in abstracts)
    markers = ["we propose", "we present", "this paper", "in this work", "results show", "our results", "we demonstrate", "future work"]
    marker_text = ", ".join(marker for marker in markers if any(marker in text.lower() for text in abstracts)) or "none of the tracked phrases"
    return (
        "Style card from this author's history: "
        f"typical length {sum(words) / len(words):.0f} words; "
        f"about {sum(sentences) / len(sentences):.1f} sentences; "
        f"first-person research language in {first_person}/{len(abstracts)} examples; "
        f"recurring discourse markers: {marker_text}."
    )


def content_card(profile: list[dict[str, Any]], limit: int = 12) -> str:
    terms = Counter(term for term in _profile_tokens(profile) if len(term) >= 4 and term not in STOPWORDS)
    top = ", ".join(term for term, _ in terms.most_common(limit))
    return f"Recurring content terms in this author's history: {top or 'none available'}."


def render_example(item: dict[str, Any], index: int, max_chars: int) -> str:
    title = " ".join(str(item.get("title", "")).split())
    abstract = " ".join(str(item.get("abstract", "")).split())
    if len(abstract) > max_chars:
        abstract = abstract[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"[Example {index}] Title: {title}\nAbstract: {abstract}"


def build_prompt(row: dict[str, Any], spec: dict[str, Any]) -> str:
    spec = normalize_spec(spec)
    if spec is None:
        raise ValueError("invalid harness spec")
    profile = list(row.get("profile") or [])
    query = str(row.get("input", ""))
    evidence = select_evidence(profile, query, spec)
    cards: list[str] = []
    mode = spec["abstraction"]["mode"]
    if mode in {"style_card", "style_content"}:
        cards.append(style_card(profile))
    if mode in {"content_card", "style_content"}:
        cards.append(content_card(profile))
    examples = [render_example(item, i + 1, spec["interface"]["max_example_chars"]) for i, item in enumerate(evidence)]
    layout = spec["interface"]["layout"]
    pieces = [
        "You are an expert academic writer.",
        "Write a complete, publication-quality abstract for the current task.",
        "Return only the final abstract, with no preface, analysis, or explanation.",
        spec["instruction"],
    ]
    if spec.get("module_guidance"):
        pieces.append("Selected module guidance:\n" + str(spec["module_guidance"]))
    if layout == "examples_only":
        pieces.extend(examples)
    elif layout == "style_then_examples":
        pieces.extend(cards)
        if examples:
            pieces.append("Historical examples from the same author:")
            pieces.extend(examples)
    elif layout == "examples_then_style":
        if examples:
            pieces.append("Historical examples from the same author:")
            pieces.extend(examples)
        pieces.extend(cards)
    else:
        pieces.extend(cards)
    if spec["graph"]["mode"] == "fallback":
        pieces.append("If retrieved examples are irrelevant, rely on the author cards and the current task instead of copying them.")
    if spec.get("router", {}).get("mode") == "topic_adaptive":
        pieces.append("Use historical examples only when they share the current topic; otherwise rely on the author cards and the task title.")
    if spec["output"]["contract"] == "structured_abstract":
        pieces.append("Cover motivation, method, findings, and significance in a coherent abstract.")
    if spec["output"]["length"] == "profile_adaptive":
        pieces.append("Use a length and sentence structure consistent with the author's historical abstracts.")
    elif spec["output"]["length"] == "concise":
        pieces.append("Keep the abstract concise while preserving the required contribution and findings.")
    pieces.append("Current task:")
    pieces.append(query)
    return "\n".join(pieces)


def chat_text(tokenizer: Any, prompt: str, system: str = "You are an expert academic writer.") -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + messages[1]["content"]


class VLLMOpenAIClient:
    """OpenAI-compatible client for a local vLLM target server.

    The evaluator deliberately talks to vLLM over HTTP so target generation
    uses the same serving path as deployment.  The server is still local to
    the allocated node; references and scoring remain entirely in this
    process.  Requests are concurrent so vLLM can use continuous batching.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 300.0,
        concurrency: int = 16,
        retries: int = 2,
        retry_wait: float = 0.5,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.timeout = float(timeout)
        self.concurrency = max(1, int(concurrency))
        self.retries = max(0, int(retries))
        self.retry_wait = max(0.0, float(retry_wait))
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def _request(self, prompt: str, payload: dict[str, Any]) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if isinstance(body.get("error"), dict):
                    raise RuntimeError(str(body["error"]))
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError(f"vLLM returned no choices for prompt: {prompt[:120]!r}")
                message = choices[0].get("message") or {}
                content = message.get("content")
                if content is None:
                    raise RuntimeError(
                        "model returned null message.content; check chat_template_kwargs "
                        "(Qwen3 requires enable_thinking=false for this contract)"
                    )
                return str(content).strip()
            except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as error:
                last_error = error
                if attempt >= self.retries:
                    break
                if self.retry_wait:
                    time.sleep(self.retry_wait * (attempt + 1))
        raise RuntimeError(f"vLLM request failed after {self.retries + 1} attempts: {last_error}") from last_error

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        *,
        system: str = "You are an expert academic writer.",
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        if not prompts:
            return []

        def generate_one(prompt: str) -> str:
            # Explicit sampling fields override a model generation_config so
            # the vLLM path matches the deterministic Transformers path for
            # evaluation and does not inherit Qwen's recommended sampling.
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": int(max_new_tokens),
                "temperature": float(temperature if do_sample else 0.0),
                "top_p": float(top_p if do_sample else 1.0),
                # Keep the frozen Answer Agent/evaluation path fully
                # deterministic.  The service also receives seed=0 below;
                # top_k=1 prevents a serving-side default from reintroducing
                # sampling when temperature is zero.
                "top_k": 1,
                # Pin the OpenAI-compatible sampling seed so a repeated
                # parent evaluation is comparable to the stored baseline even
                # when vLLM receives requests in a different batch order.
                "seed": 0,
                "repetition_penalty": 1.0,
                "n": 1,
                "stream": False,
            }
            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
            return self._request(prompt, payload)

        workers = min(self.concurrency, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(generate_one, prompts))


def generate_batch(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
    system: str = "You are an expert academic writer.",
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    if not prompts:
        return []
    if isinstance(model, VLLMOpenAIClient):
        return model.generate_batch(
            prompts,
            max_new_tokens,
            system=system,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    import torch
    texts = [chat_text(tokenizer, prompt, system=system) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip() for row in output]


def _stage(spec: dict[str, Any]) -> str:
    return str(normalize_spec(spec)["reasoning"]["stages"][0])


def generate_records(
    tokenizer: Any,
    model: Any,
    records: list[tuple[int, dict[str, Any], dict[str, Any]]],
    batch_size: int = 8,
) -> dict[int, str]:
    """Generate final outputs for records with possibly different specs.

    Records are grouped by workflow stage, but prompts remain user-specific.
    This lets a population of evolving harnesses share GPU batches.
    """
    results: dict[int, str] = {}
    groups: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record in records:
        groups[_stage(record[2])].append(record)
    for stage, group in groups.items():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            base_prompts = [build_prompt(row, spec) for _, row, spec in chunk]
            max_input = max(int(normalize_spec(spec)["budget"]["max_input_tokens"]) for _, _, spec in chunk)
            max_new = max(int(normalize_spec(spec)["budget"]["max_new_tokens"]) for _, _, spec in chunk)
            if stage == "direct":
                outputs = generate_batch(tokenizer, model, base_prompts, max_input, max_new)
            elif stage == "draft_revision":
                drafts = generate_batch(tokenizer, model, [prompt + "\nDraft the abstract now." for prompt in base_prompts], max_input, max_new)
                revision_prompts = [
                    prompt + "\n\nFirst draft:\n" + draft + "\n\nRevise the draft to satisfy the task and author style. Return only the final abstract."
                    for prompt, draft in zip(base_prompts, drafts)
                ]
                outputs = generate_batch(tokenizer, model, revision_prompts, max_input, max_new)
            else:
                drafts = generate_batch(tokenizer, model, [prompt + "\nDraft the abstract now." for prompt in base_prompts], max_input, max_new)
                critique_prompts = [
                    prompt + "\n\nDraft:\n" + draft + "\n\nList only the three most important content or style fixes needed before finalizing."
                    for prompt, draft in zip(base_prompts, drafts)
                ]
                critiques = generate_batch(tokenizer, model, critique_prompts, max_input, max(96, min(192, max_new)))
                final_prompts = [
                    prompt + "\n\nDraft:\n" + draft + "\n\nCritique:\n" + critique + "\n\nWrite the final abstract only."
                    for prompt, draft, critique in zip(base_prompts, drafts, critiques)
                ]
                outputs = generate_batch(tokenizer, model, final_prompts, max_input, max_new)
            for (index, _, _), output in zip(chunk, outputs):
                results[index] = output
    return results


def extract_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    return None


MUTABLE_SPEC_PATHS = {
    "evidence.policy",
    "evidence.k",
    "evidence.query_mode",
    "abstraction.mode",
    "interface.layout",
    "interface.max_example_chars",
    "router.mode",
    "router.threshold",
    "router.fallback",
    "reasoning.stages",
    "output.contract",
    "output.length",
    "graph.mode",
    "budget.max_calls",
    "budget.max_input_tokens",
    "budget.max_new_tokens",
    "instruction",
}


def _flatten_spec(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a normalized spec for safe, interpretable mutation checks."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_spec(child, path))
        return result
    return {prefix: value}


def spec_diff(parent: dict[str, Any], child: dict[str, Any]) -> list[dict[str, Any]]:
    """Return changed mutable fields between two normalized harnesses."""
    left = _flatten_spec(normalize_spec(parent) or {})
    right = _flatten_spec(normalize_spec(child) or {})
    changes: list[dict[str, Any]] = []
    for path in sorted(set(left) | set(right)):
        if path not in MUTABLE_SPEC_PATHS:
            continue
        if left.get(path) != right.get(path):
            changes.append({"path": path, "before": left.get(path), "after": right.get(path)})
    return changes


def apply_spec_patch(parent: dict[str, Any], patch: Any) -> dict[str, Any] | None:
    """Apply one JSON patch-like ``path``/``value`` edit to a harness.

    LLMs are much more reliable at proposing one path/value pair than a
    complete nested object.  The function accepts either dotted paths or JSON
    pointer paths and rejects unknown fields before normalization.
    """
    if isinstance(patch, dict) and isinstance(patch.get("patch"), dict):
        patch = patch["patch"]
    if not isinstance(patch, dict):
        return None
    path = patch.get("path")
    if not isinstance(path, str):
        return None
    path = path.strip().strip("/").replace("/", ".")
    if path not in MUTABLE_SPEC_PATHS:
        return None
    if "value" not in patch:
        return None
    candidate = deepcopy(parent)
    target: Any = candidate
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(target, dict):
            return None
        target = target.setdefault(part, {})
    if not isinstance(target, dict):
        return None
    target[parts[-1]] = deepcopy(patch["value"])
    normalized = normalize_spec(candidate)
    if normalized is None:
        return None
    return normalized


def summarize_user_outputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row_metrics(row) for row in rows]
    if not metrics:
        return {"n": 0}
    mean = {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}
    low = sum(item["rougeL"] < 0.15 for item in metrics) / len(metrics)
    length_ratio = sum(item["prediction_tokens"] / max(1.0, item["target_tokens"]) for item in metrics) / len(metrics)
    traces = []
    for row, metric in zip(rows, metrics):
        prediction = " ".join(str(row.get("prediction", "")).split())
        traces.append({
            "task": task_title(str(row.get("input", "")))[:220],
            "prediction": prediction[:600],
            "rougeL": round(metric["rougeL"], 4),
            "bleu": round(metric["bleu"], 4),
            "meteor": round(metric["meteor"], 4),
            "title_coverage": round(title_coverage(str(row.get("input", "")), prediction), 4),
            "prediction_tokens": int(metric["prediction_tokens"]),
            "target_tokens": int(metric["target_tokens"]),
        })
    traces.sort(key=lambda item: (item["rougeL"], item["title_coverage"], item["task"]))
    return {
        "n": len(metrics),
        "mean_rouge1": round(mean["rouge1"], 5),
        "mean_rouge2": round(mean["rouge2"], 5),
        "mean_rougeL": round(mean["rougeL"], 5),
        "mean_bleu": round(mean["bleu"], 5),
        "mean_meteor": round(mean["meteor"], 5),
        "low_rougeL_fraction": round(low, 5),
        "prediction_target_length_ratio": round(length_ratio, 5),
        "mean_prediction_tokens": round(mean["prediction_tokens"], 1),
        "mean_title_coverage": round(sum(item["title_coverage"] for item in traces) / len(traces), 5),
        # Reflection sees execution traces and task titles, but never the
        # reference abstract.  The lowest-scoring examples make failure
        # diagnosis concrete instead of prompting a generic revision.
        "worst_traces": traces[:4],
    }
