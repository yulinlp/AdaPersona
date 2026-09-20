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

The outer Code Agent is always Qwen3.8 and edits complete Python source. It is
not available inside the runtime harness. The paired experiment separately
uses Qwen2.5-7B and Qwen3.8 as frozen Answer Models.
The runtime QA broker clamps generation to deterministic temperature `0`,
`top_p=1`, `top_k=1`, and seed `0`; stochasticity belongs to the outer
evolution agent, not official harness scoring.

## Search

Each of 10 rounds chooses either macro strategy exploration or micro repair,
with at most three candidate slots total. There is no forced alternation.
The archive-beam search allocates branches over high-scoring, diverse, and
under-explored parents. The Code Agent first proposes distinct hypotheses and
then writes complete source for each hypothesis. Invalid code, duplicate ASTs,
runtime failures, and service failures are recorded in the per-user failure
bank. The failure bank, history, and archived source/diffs are fed back to
later proposals for that same user only.

The starting program is selected among five seeds on independent historical
selection tasks. Candidates require zero errors, strict fitting improvement and
no historical-selection regression. Both partitions are rechecked on candidate
and parent before acceptance. For text generation the objective is:

`0.25*ROUGE-1 + 0.35*ROUGE-L + 0.20*BLEU + 0.20*METEOR`.

All reports use a 0--100 display scale. Exact metric versions and weights are
stored in every run's `run_config.json`.

## Test-time profile-only protocol

For strict user split evaluation, the official test user is unseen during
training. Its historical profile is split into leave-one-out pseudo tasks:

- `profile_adaptation.jsonl` is used for per-user evolution;
- `profile_selection.jsonl` selects the initial seed and gates candidate acceptance;
- all selection examples are removed from fitting profiles and diagnostics;
- `test.jsonl` is scored independently after seed and each committed round.

The pseudo references are historical profile abstracts. They are allowed
profile evidence, not the official test target. The final scorer uses the
official test target only outside the harness to calculate metrics.

The same profile-only adapter supports the original LaMP user split for tasks
2, 3, 4, and 5 only. LaMP-2 optimizes exact label accuracy;
LaMP-3 optimizes rating closeness (1 - MAE/4); LaMP-4 and LaMP-5
reuse the text-overlap objective above. LaMP-QA and LaMP-1/6/7 are outside
this protocol. See `scripts/lamp_tasks.py` and
`scripts/evaluate_lamp_test.py`.
For per-round visibility, `scripts/monitor_lamp_test.py` is a separate
side-channel scorer; it cannot affect acceptance or Code Agent prompts.

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
