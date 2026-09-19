# V5 per-user Python harness RSI

The canonical launcher is `scripts/run_per_user_evo.sh`, backed by
`scripts/evolve_per_user.py` and `scripts/code_evolution.py`.

## Isolation and data boundary

The runner groups rows by `user_id` and evolves each user's harness
independently. No cross-user score, profile, failure bank, or source is used in
a user's acceptance decision. The recommended generalization protocol is
strict user-split test-time profile-only adaptation:

```text
historical profile -> leave-one-out pseudo tasks -> per-user harness evolution
historical profile holdout -> optional seed/current selection
official test task -> one frozen harness call -> outer scorer
```

The official test target is never sent to the harness or Code Agent. The
historical profile abstracts used as pseudo references are explicitly recorded
as profile-only self-supervision. This is adaptation using allowed user history,
not model-parameter training.

## V5 search

The default is 10 iterations, four proposals per operation, and archive-beam
search with beam width four, archive size 64, and four islands. Each iteration
contains:

1. `macro_strategy`: propose a materially different black-box workflow;
2. `micro_repair`: repair a measured weakness while preserving the useful core.

The Code Agent proposes hypotheses and writes executable Python source. Source
is AST-validated, executed in a fresh process, and evaluated on that user's
adaptation rows. Candidate failures and successes are appended to that user's
failure bank and shown to later proposals. An accepted candidate must have zero
execution errors and strictly improve the weighted objective; it is then
confirmed against a fresh parent execution.

## Models

Default:

- Code Agent: Qwen3.8-27B at `gpu02:18012`;
- Answer Agent: frozen Qwen2.5-7B-Instruct at `gpu01:8000`;
- optional embeddings: frozen Qwen3-Embedding-0.6B at `gpu01:18013`.

The same runner can use Qwen3.8 for both roles:

```bash
AGENT_URL=http://gpu02:18012/v1 AGENT_MODEL=Qwen/Qwen3.8-27B \
QA_URL=http://gpu02:18012/v1 QA_MODEL=Qwen/Qwen3.8-27B \
bash scripts/run_per_user_evo.sh
```

The QA model is black-box from the harness's perspective. The runner renders
the selected model and context capacity into the Code Agent contract. With an
8K Code Agent endpoint, the prompt fitter compresses literature/history detail
while retaining the complete parent source and measured evidence.

## Metrics

Candidate selection uses the exact weighted objective:

`0.25*ROUGE-1 + 0.35*ROUGE-L + 0.20*BLEU + 0.20*METEOR`.

The project uses rouge-score 0.1.2, NLTK 3.9.2 METEOR with WordNet, and
sacrebleu 2.5.1 sentence BLEU. Reports expose ROUGE-1, ROUGE-L, BLEU, METEOR,
and the weighted score on a 0--100 scale. Title coverage is diagnostic only.

## Artifacts and restart behavior

The runner checkpoints after each operation. A run stores `run_config.json`,
`source_snapshot/`, `progress.json`, `user_scores.json`, and per-user source,
prediction, confirmation, history, archive, and failure-bank files. A source or
protocol hash mismatch requires a fresh output directory. Service errors are
visible and retried a bounded number of times; an operation with no usable
candidate does not advance.
