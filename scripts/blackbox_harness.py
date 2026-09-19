"""Runtime and validation helpers for the code-level black-box harness.

The only contract exposed to an evolved harness is::

    def run(row: dict, qa: QA) -> str

``row`` contains the current task and the user's historical profile.  ``qa``
is the frozen QA-model client and is the only supported way for the harness to
call an LLM.  The evolver is deliberately not represented as a runtime
module: it edits this file-level program as a whole.

This first version intentionally exposes no logits, hidden states, gradients,
model weights, or external tools.  It is therefore compatible with an
ordinary OpenAI-compatible vLLM endpoint only.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    from longlamp_rsi import VLLMOpenAIClient
except ModuleNotFoundError:  # pragma: no cover - package-style import
    from scripts.longlamp_rsi import VLLMOpenAIClient


HARNESS_INTERFACE_VERSION = "blackbox-v2"


class QA(Protocol):
    """The only runtime LLM interface available to an evolved harness."""

    def embed(self, texts: list[str], *, instruction: str | None = None) -> list[list[float]]:
        ...

    def generate(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        ...

    def generate_many(
        self,
        prompts: list[str],
        *,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        ...


class VLLMQA:
    """Thin black-box adapter around the local OpenAI-compatible vLLM server."""

    def __init__(self, client: VLLMOpenAIClient, max_calls: int = 8,
                 embedding_client=None, deterministic: bool = True) -> None:
        self.client = client
        self._embedding_client = embedding_client
        self.max_calls = max(1, int(max_calls))
        # Evolution-time Code Agent calls may explore stochastically, but the
        # answer/evaluation broker must be reproducible.  A harness may pass
        # sampling kwargs through the public signature for compatibility, but
        # they are clamped here before reaching the frozen QA model.
        self.deterministic = bool(deterministic)
        self.calls = 0
        self.embedding_texts = 0
        self.embedding_chars = 0
        # Target-text-free execution trace exposed to the outer evolver.  It
        # is intentionally kept on the QA proxy rather than in the harness
        # contract, so the evolved program does not need to know about logs.
        self.trace: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def embed(self, texts: list[str], *, instruction: str | None = None) -> list[list[float]]:
        if self._embedding_client is None:
            raise RuntimeError('Embedding service is not configured for this run')
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError('embed expects list[str]')
        with self._lock:
            self.embedding_texts += len(texts)
            self.embedding_chars += sum(len(t) for t in texts)
            if self.embedding_texts > 4096 or self.embedding_chars > 2_000_000:
                raise RuntimeError('embedding budget exceeded (4096 texts / 2000000 characters)')
        vectors = self._embedding_client.embed(texts, instruction=instruction)
        self.trace.append({'kind': 'embedding', 'texts_count': len(texts),
                           'instruction': instruction,
                           'dimension': len(vectors[0]) if vectors else 0})
        return vectors

    def _reserve(self, amount: int) -> None:
        with self._lock:
            if self.calls + amount > self.max_calls:
                raise RuntimeError(
                    f"harness exceeded QA call budget ({self.max_calls})"
                )
            self.calls += amount

    def generate(
        self,
        prompt: str,
        *,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        if self.deterministic:
            temperature, top_p = 0.0, 1.0
        self._reserve(1)
        output = self.client.generate_batch(
            [str(prompt)],
            max_new_tokens=max(1, int(max_tokens)),
            system=str(system),
            do_sample=float(temperature) > 0.0,
            temperature=float(temperature),
            top_p=float(top_p),
        )[0]
        self.trace.append({
            "prompt": str(prompt),
            "system": str(system),
            "response": str(output),
            "max_tokens": int(max_tokens),
        })
        return output

    def generate_many(
        self,
        prompts: list[str],
        *,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        prompts = [str(prompt) for prompt in prompts]
        if not prompts:
            return []
        if self.deterministic:
            temperature, top_p = 0.0, 1.0
        self._reserve(len(prompts))
        outputs = self.client.generate_batch(
            prompts,
            max_new_tokens=max(1, int(max_tokens)),
            system=str(system),
            do_sample=float(temperature) > 0.0,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        self.trace.extend({
            "prompt": prompt,
            "system": str(system),
            "response": str(output),
            "max_tokens": int(max_tokens),
        } for prompt, output in zip(prompts, outputs))
        return outputs


class HarnessValidationError(ValueError):
    """Raised when an agent-produced harness cannot be safely executed."""


_SAFE_IMPORT_ROOTS = {
    "numpy",
    "scipy",
    "sklearn",
    "networkx",
    "collections",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "string",
}
_FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "pathlib",
    "socket",
    "subprocess",
    "sys",
    "requests",
    "urllib",
    "http",
    "shutil",
    "ctypes",
    "pickle",
}
_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
}


def code_fingerprint(code: str) -> str:
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()[:16]


def strip_code_fence(value: str) -> str:
    text = str(value or "").strip()
    # Agents sometimes add one short sentence before/after the fence even
    # when asked for source only. Extract the first complete Python fence so
    # that harmless transport prose cannot become a syntax error on line 1.
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.I | re.S)
    if match:
        return match.group(1).strip()
    # A serving response can be cut at a natural end-of-source boundary
    # without emitting the closing fence.  The content after the opening
    # fence is still a complete candidate; do not turn it into a needless
    # compiler-repair request.
    opening = re.match(r"^```(?:python|py)?\s*\n?", text, re.I)
    if opening:
        remainder = text[opening.end():]
        if remainder.rstrip().endswith("```"):
            remainder = remainder.rstrip()[:-3]
        return remainder.strip()
    return text


def _import_guard(name: str, globals: dict[str, Any] | None = None, locals: dict[str, Any] | None = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    root = str(name).split(".", 1)[0]
    if level or root not in _SAFE_IMPORT_ROOTS:
        raise ImportError(f"harness import is not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    names = {
        "abs", "all", "any", "bool", "dict", "enumerate", "Exception",
        "filter", "float", "frozenset", "hasattr", "int", "isinstance",
        "iter", "KeyError", "len", "list", "map", "max", "min", "next",
        "print", "range", "reversed", "round", "set", "sorted", "str",
        "sum", "tuple", "TypeError", "ValueError", "zip",
        "object", "type", "super", "property", "staticmethod", "classmethod",
        "__build_class__", "RuntimeError", "NameError", "IndexError",
        "AttributeError", "StopIteration", "NotImplementedError", "AssertionError",
        "bytes", "bytearray", "slice", "repr", "format", "pow", "divmod",
        "ord", "chr", "callable", "issubclass",
    }
    result = {name: getattr(builtins, name) for name in names}
    result["__import__"] = _import_guard
    return result


def validate_harness_code(code: str) -> list[str]:
    """Return validation errors; an empty list means the code is executable."""
    errors: list[str] = []
    text = strip_code_fence(code)
    if not text:
        return ["empty harness code"]
    try:
        tree = ast.parse(text, filename="harness.py", mode="exec")
    except SyntaxError as error:
        return [f"syntax error: {error}"]

    run_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"]
    if len(run_functions) != 1:
        errors.append("harness must define exactly one top-level def run(row, qa)")
    elif isinstance(run_functions[0], ast.AsyncFunctionDef):
        errors.append("run must be synchronous, not async")
    else:
        args = run_functions[0].args
        if len(args.posonlyargs + args.args) != 2 or args.vararg or args.kwarg or args.kwonlyargs:
            errors.append("run must accept exactly two positional arguments: row, qa")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _SAFE_IMPORT_ROOTS:
                    errors.append(f"import is not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0]
            if node.level or root not in _SAFE_IMPORT_ROOTS:
                errors.append(f"import is not allowed: {node.module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                errors.append(f"call is not allowed: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_CALLS:
                errors.append(f"call is not allowed: {node.func.attr}")
        elif isinstance(node, ast.Name) and node.id in {"__builtins__", "__loader__", "__spec__"}:
            errors.append(f"runtime escape is not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr == 'mro'):
            errors.append(f"runtime introspection is not allowed: {node.attr}")
    return sorted(set(errors))


class CompiledHarness:
    """Compile one candidate once and expose its fixed ``run`` entry point."""

    def __init__(self, code: str) -> None:
        self.code = strip_code_fence(code)
        errors = validate_harness_code(self.code)
        if errors:
            raise HarnessValidationError("; ".join(errors))
        namespace: dict[str, Any] = {
            "__builtins__": _safe_builtins(),
            "__name__": "evolved_harness",
        }
        try:
            exec(compile(self.code, "harness.py", "exec"), namespace, namespace)
        except Exception as error:  # pragma: no cover - agent code dependent
            raise HarnessValidationError(f"harness import failed: {error}") from error
        run = namespace.get("run")
        if not callable(run):
            raise HarnessValidationError("harness run is not callable")
        self._run: Callable[[dict[str, Any], QA], Any] = run

    def run(self, row: dict[str, Any], qa: QA) -> str:
        value = self._run(row, qa)
        if not isinstance(value, str):
            raise TypeError("run must return str, not " + type(value).__name__)
        output = value.strip()
        if not output:
            raise RuntimeError("harness returned an empty string")
        return output


def runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    """Remove all evaluation-only fields before code sees a row."""
    return {
        "user_id": str(row.get("user_id", "")),
        "sample_id": str(row.get("sample_id", "")),
        "input": str(row.get("input", "")),
        "profile": deepcopy(list(row.get("profile") or [])),
    }


SEED_HARNESS_CODE = r'''"""Seed black-box personalized writing harness."""
from collections import Counter
import math
import re


def _tokens(text):
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text).lower())


def _title(text):
    match = re.search(r'title\s+["“](.*?)["”]', str(text), re.I | re.S)
    return " ".join(match.group(1).split()) if match else str(text).strip()


def _bm25(profile, query, k=4):
    if not profile:
        return []
    q = Counter(_tokens(query))
    docs = [_tokens(str(item.get("title", "")) + " " + str(item.get("abstract", ""))) for item in profile]
    df = Counter(term for doc in docs for term in set(doc))
    avgdl = sum(len(doc) for doc in docs) / max(1, len(docs))
    scored = []
    for index, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0
        for term, qtf in q.items():
            if not tf[term]:
                continue
            idf = math.log(1.0 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + 0.9 * (0.25 + 0.75 * len(doc) / max(1.0, avgdl))
            score += idf * tf[term] * 1.9 / denom * qtf
        scored.append((score, index, profile[index]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:max(0, int(k))]]


def _recent(profile, k=3):
    return sorted(
        profile,
        key=lambda item: (int(item.get("year", 0)), str(item.get("id", ""))),
        reverse=True,
    )[:max(0, int(k))]


def _style_card(profile):
    abstracts = [str(item.get("abstract", "")).strip() for item in profile if item.get("abstract")]
    if not abstracts:
        return "No historical style evidence is available."
    lengths = [len(_tokens(text)) for text in abstracts]
    sentences = [max(1, len(re.findall(r"[.!?]+", text))) for text in abstracts]
    markers = [
        "we propose", "we present", "this paper", "in this work",
        "results show", "we demonstrate", "experimental results",
    ]
    recurring = [marker for marker in markers if sum(marker in text.lower() for text in abstracts) >= max(1, len(abstracts) // 10)]
    return (
        "Historical style prior: typical length about %.0f words; about %.1f sentences; "
        "recurring discourse markers: %s. Use it softly and never copy historical content."
        % (sum(lengths) / len(lengths), sum(sentences) / len(sentences), ", ".join(recurring) or "none")
    )


def _render(item, index):
    title = " ".join(str(item.get("title", "")).split())
    abstract = " ".join(str(item.get("abstract", "")).split())[:1200]
    return "[Historical example %d]\nTitle: %s\nAbstract: %s" % (index, title, abstract)


def run(row, qa):
    profile = list(row.get("profile") or [])
    task = str(row.get("input", ""))
    query = _title(task)
    topical = _bm25(profile, query, 4)
    recent = _recent(profile, 2)
    selected = []
    seen = set()
    for item in topical + recent:
        key = str(item.get("id", item.get("title", "")))
        if key not in seen:
            selected.append(item)
            seen.add(key)
    evidence = "\n\n".join(_render(item, index + 1) for index, item in enumerate(selected))
    prompt = """Write a complete publication-quality abstract for the current task.
Return only the abstract, with no analysis or preface.
Preserve the concrete entities and requested topic in the current task. Use the
historical evidence only as a soft prior for the author's organization, style,
and level of detail; do not copy unrelated content or invent unsupported facts.

%s

Historical examples:
%s

Current task:
%s""" % (_style_card(profile), evidence or "No relevant examples were retrieved.", task)
    return qa.generate(
        prompt,
        system="You are an expert academic writer who follows the current task exactly.",
        max_tokens=256,
        temperature=0.0,
    )
'''


def save_code(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strip_code_fence(code))
