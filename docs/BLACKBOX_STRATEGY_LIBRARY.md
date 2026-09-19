# Black-box strategy library for Personalized Harness RSI

This is a reference library, not a module specification.  An evolved harness
may combine, replace, or ignore these ideas.  The only runtime contract is
`run(row, qa) -> str`, and all runtime model calls must go through the frozen
QA client exposed as `qa`.

This list is non-exhaustive and has no preference ordering. A simple direct
call and a complex graph workflow are equally legitimate hypotheses. Evidence
from the current user's training comparisons, not list order or complexity,
should motivate choices. Title overlap is only a diagnostic, not an objective.

## Context and memory construction

- Full-history and recency memory: use all history, a recent window, or a
  temporally weighted mixture.
- Lexical, semantic, hybrid, and diversity retrieval: rewrite the task query,
  retrieve by title/content similarity, mix topical and recent evidence, and
  rerank for coverage rather than simply taking the top-k.
- Hierarchical memory: maintain raw episodes, short summaries, long-term
  user facts, and task-conditioned summaries at different compression levels.
- Profile and preference memory: construct style, content, preference,
  negative-preference, or task-specific cards from the supplied history.
- Graph memory: build lightweight entity/topic/relation/co-occurrence graphs
  from historical items; retrieve a subgraph, a community, or a path related
  to the current task, then render it as text.  A graph can be implemented
  with ordinary Python data structures; no graph database is required.
- Multi-view memory: keep separate views for style, content, structure,
  terminology, chronology, and user preferences, then choose or combine views
  per task.
- Memory conflict handling: prefer recent evidence, stable evidence, or a
  consensus of compatible historical items when user history is inconsistent.

## Prompt and context construction

- Place the current task before or after evidence; label examples clearly;
  separate style instructions from content evidence.
- Use positive examples, negative examples, contrastive pairs, or a compact
  profile instead of blindly concatenating history.
- Adapt context size and detail to task difficulty, profile size, and evidence
  relevance.  Drop irrelevant evidence instead of filling a fixed top-k.
- Use explicit output contracts, planning hints, checklists, or structured
  intermediate notes when the task benefits from them.

## Black-box generation control

- Direct generation, draft-then-revise, critique-then-revise, and multi-pass
  self-refinement using the same QA endpoint.
- Generate several candidates and select or rerank them using task fidelity,
  user style/profile consistency, coverage, or a QA-model judge.
- Use preference-guided selection with an explicit preference checklist or
  matrix represented as text.  It may score complete candidates; it must not
  assume token logits or hidden states.
- Allocate a small or large generation budget based on profile evidence,
  task complexity, or uncertainty estimated from black-box outputs.

## Strategy composition and routing

- Route different task types WITHIN THIS USER to different workflows when
  justified by evidence; do not implement a global user router or meta-policy.
- Compose a memory strategy with a generation strategy, for example graph
  memory plus candidate selection, or profile compression plus revision.
- Keep cheap fallbacks for empty or irrelevant profiles and for failed extra
  calls.  A fallback should preserve the current task rather than copy memory.

## Deliberately excluded from version 1

The ordinary vLLM API does not expose the capabilities needed for these, so
the first black-box version must not pretend to implement them:

- hidden-state activation steering such as StyleVector or GLASS;
- token-level contrastive/logit decoding such as exact CoS;
- direct access to logits, gradients, KV cache, or model weights;
- LoRA, PEFT, fine-tuning, learned retrievers, or online weight updates;
- direct calls to a named model, unprovided external APIs/tools, or internet
  from the runtime harness. If the selected QA endpoint happens to be Qwen3.8,
  the harness still sees only the public `qa` interface.

The default experiment uses Qwen3.8-27B as the outer Code Agent and frozen
Qwen2.5-7B-Instruct as QA. A Qwen3.8/Qwen3.8 comparison is supported as a
separate explicitly labelled protocol; model choice does not change the
`run(row, qa) -> str` contract.
