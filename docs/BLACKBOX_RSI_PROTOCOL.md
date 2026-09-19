# AdaPersona V5 black-box RSI protocol

The current implementation is the per-user V5 code-evolution protocol. Older
global, modular, and JSON-spec runners are not part of the repository.

The only harness contract is:

```python
def run(row, qa) -> str:
    ...
```

`row` contains the current target-free task and the user's historical profile.
`qa` exposes only `generate`, `generate_many`, and `embed`. The harness cannot
access files, networks, subprocesses, model weights, gradients, logits, or
hidden states. It may implement retrieval, profile cards, graph-like memory,
candidate generation, revision, or another in-memory black-box workflow.

The outer Code Agent edits complete Python source. It is not available inside
the runtime harness. The default answer model is Qwen2.5-7B-Instruct, but the
runner accepts an arbitrary served QA model through `--qa-model`; this supports
the Qwen3.8/Qwen3.8 comparison without changing the interface.

## Search

Each iteration runs a macro strategy operation and a micro repair operation.
The archive-beam search allocates branches over high-scoring, diverse, and
under-explored parents. The Code Agent first proposes distinct hypotheses and
then writes complete source for each hypothesis. Invalid code, duplicate ASTs,
runtime failures, and service failures are recorded in the per-user failure
bank. The failure bank, history, and archived source/diffs are fed back to
later proposals for that same user only.

An accepted candidate must have zero errors and a strictly higher weighted score
than its parent. It is executed again together with a parent recheck before the
state advances. The objective is:

`0.25*ROUGE-1 + 0.35*ROUGE-L + 0.20*BLEU + 0.20*METEOR`.

All reports use a 0--100 display scale. Exact metric versions and weights are
stored in every run's `run_config.json`.

## Test-time profile-only protocol

For strict user split evaluation, the official test user is unseen during
training. Its historical profile is split into leave-one-out pseudo tasks:

- `profile_adaptation.jsonl` is used for per-user evolution;
- `profile_selection.jsonl` optionally chooses seed versus evolved incumbent;
- `test.jsonl` is evaluated only after the harness is frozen.

The pseudo references are historical profile abstracts. They are allowed
profile evidence, not the official test target. The final scorer uses the
official test target only outside the harness to calculate metrics.

## Runtime isolation

Every sample executes in a fresh worker process. The broker enforces at most
eight generation calls and the embedding budget per sample. The worker gets a
target-free copy of the row and a JSON-RPC-like QA facade. These guards are
defense in depth for research code, not a hardened adversarial sandbox.

## Reproducibility artifacts

Each V5 run stores its source snapshot, service/model configuration, data hash,
candidate source, predictions, confirmation rechecks, failure bank, archive,
history, and progress. A run directory must not be resumed after changing the
protocol or source; use a new output directory.
