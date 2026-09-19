# AdaPersona

AdaPersona is a black-box, per-user personalized-harness RSI system for
LongLaMP abstract generation.  The current protocol is V5: a separate Python
harness is evolved for each user, while the harness keeps the fixed interface
`run(row, qa) -> str`.

The runtime is model-agnostic at the interface level.  All model calls made by
an evolved harness go through the frozen `qa` broker; the outer Code Agent
rewrites complete harness source during evolution.  The default experiment is
Qwen3.8-27B Code Agent + Qwen2.5-7B Answer Agent, with a Qwen3.8/Qwen3.8
variant supported by the same CLI.

## Strict user-split test-time adaptation

The reproducible protocol is:

1. Prepare the disjoint LongLaMP user split with
   `scripts/prepare_longlamp_abstract_user_pilot.py`.
2. Build test-user profile-only leave-one-out tasks with
   `scripts/prepare_profile_only_tta.py`.
3. Evolve one harness per test user using only `profile_adaptation.jsonl`.
4. Optionally select seed/current using `profile_selection.jsonl`.
5. Run the frozen selected harness once on `test.jsonl` with
   `scripts/evaluate_profile_only_test.py`.

The official test target is available only to the outer scorer.  It is not
passed to a harness, Code Agent, failure bank, or selection step.

Example setup:

```bash
python scripts/prepare_longlamp_abstract_user_pilot.py \
  --output data/experiments/longlamp_abstract_user_rsi
python scripts/prepare_profile_only_tta.py \
  --output data/experiments/longlamp_abstract_user_rsi_tta
```

Run the default V5 search:

```bash
TRAIN=data/experiments/longlamp_abstract_user_rsi_tta/profile_adaptation.jsonl \
OUTPUT_DIR=data/experiments/longlamp_abstract_user_rsi_tta/runs/profile_only_v5_i10_b4 \
bash scripts/run_per_user_evo.sh
```

The shell wrapper also accepts `AGENT_URL`, `AGENT_MODEL`, `QA_URL`, and
`QA_MODEL`.  For example, both evolution and answering can use Qwen3.8:

```bash
AGENT_URL=http://gpu02:18012/v1 AGENT_MODEL=Qwen/Qwen3.8-27B \
QA_URL=http://gpu02:18012/v1 QA_MODEL=Qwen/Qwen3.8-27B \
bash scripts/run_per_user_evo.sh
```

Then select and score:

```bash
python scripts/select_profile_only_harness.py \
  --run-dir data/experiments/longlamp_abstract_user_rsi_tta/runs/profile_only_v5_i10_b4 \
  --selection data/experiments/longlamp_abstract_user_rsi_tta/profile_selection.jsonl

python scripts/evaluate_profile_only_test.py \
  --run-dir data/experiments/longlamp_abstract_user_rsi_tta/runs/profile_only_v5_i10_b4 \
  --output-dir data/experiments/longlamp_abstract_user_rsi_tta/runs/official_test_scores
```

For a live view of official test performance after every completed evolution
operation, run the independent hidden-test monitor:

    python scripts/monitor_hidden_test.py \
      --run-dir data/experiments/longlamp_abstract_user_rsi_tta/runs/profile_only_v5_i10_b4 \
      --test data/experiments/longlamp_abstract_user_rsi_tta/test.jsonl \
      --output-dir data/experiments/longlamp_abstract_user_rsi_tta/shadow_test_monitor

It writes only metrics outside the evolution run directory. Hidden targets and
scores never enter candidate selection, the failure bank, or Code Agent prompts.

## V5 search contract

Each operation has two families: `macro_strategy` proposes a substantially
different workflow, and `micro_repair` repairs a measured weakness.  The
archive-beam policy keeps diverse evaluated parents instead of following only
one chain.  Candidates are executable Python source, validated and run in
isolated processes.  A candidate must have zero execution errors and strictly
improve the per-user weighted objective to become the incumbent; accepted
candidates are rechecked against a fresh parent execution.

The objective is reported on a 0--100 scale:

`0.25 ROUGE-1 + 0.35 ROUGE-L + 0.20 BLEU + 0.20 METEOR`.

See [docs/PER_USER_RSI.md](docs/PER_USER_RSI.md) for the full protocol and
[docs/BLACKBOX_STRATEGY_LIBRARY.md](docs/BLACKBOX_STRATEGY_LIBRARY.md) for the
non-binding strategy reference given to the Code Agent.

## Baselines and tests

`scripts/benchmark_longlamp_nontraining.py` contains same-backbone
Full-context, BM25-RAG, PAG/profile-summary, and CoT baselines.  The test suite
is run with:

```bash
python -m unittest discover -s tests -q
```

Install metric dependencies from `requirements-evolution.txt`.  Local model
servers and datasets are intentionally excluded from Git.
