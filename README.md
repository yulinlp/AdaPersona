# AdaPersona

**Personalization by evolving a user's inference program, without updating model weights.**

## Motivation

People differ in interests, writing style, and preferences. One fixed retrieval-and-prompt recipe need not work equally well for every user. AdaPersona asks whether an LLM can use a user's historical examples to discover a better *inference harness*: executable code that organizes history, retrieves evidence, constructs prompts, and combines black-box model calls.

The research hypothesis is that per-user program search can improve personalization over fixed inference recipes. This repository implements that hypothesis; it does not claim state-of-the-art performance or established generalization from a small pilot.

## Method

Each user has an independent Python program with one interface:

```python
def run(row, qa) -> str:
    # row: current input and historical profile; no reference answer
    # qa: generate(), generate_many(), embed()
    ...
```

There are no mandatory internal modules. The Code Agent may implement lexical/semantic retrieval, profile summaries, graph-based memory, prompt construction, planning, revision, or candidate selection. The [strategy library](docs/BLACKBOX_STRATEGY_LIBRARY.md) provides optional inspiration, not a required architecture. White-box activation editing and custom decoding hooks are outside the API contract.

For each user:

1. Partition usable historical examples deterministically into up to eight fitting tasks and up to four selection tasks (smaller profiles use smaller sets). All selection examples are removed from fitting profiles as well as fitting targets. Each task also excludes its own historical source item. Historical outputs are self-supervised targets, not official test answers.
2. Evaluate the retrieval seed, Full Context, RAG, PAG and CoT on these historical tasks. Choose the best seed by selection score, breaking ties with fitting score, and retain the other fitting-evaluated seeds in the archive. No official test score is involved.
3. For each of **10 rounds**, the Code Agent chooses **one** operation: `macro_strategy` (change strategy) or `micro_repair` (refine implementation). There is no alternation schedule or operation quota.
4. Propose at most **3 candidate programs total per round**. Archive-beam search retains alternative lineages rather than following only accepted programs. By default one slot is reserved for the incumbent; the rest explore archive parents when available. `--incumbent-slots` exposes this search-policy choice.
5. Validate and execute candidates in disposable processes. The failure bank records fitting errors and regressions only. Rank candidates by historical selection score, then fitting score. Acceptance requires zero errors, a strict fitting improvement, no historical-selection regression, and fresh candidate/parent rechecks on both partitions. Otherwise keep the incumbent. Selection examples, predictions and detailed scores are never passed to the Code Agent.
6. Independently evaluate the incumbent on the user's official test task after the seed and every committed round.

Ten rounds provide at most 30 candidate slots, not 30 accepted improvements. Seed evaluations, syntax-repair requests, smoke executions and confirmation executions consume additional compute. This is recursive **code** improvement, not gradient training. No meta-policy is trained in these experiments.

For LongLaMP, historical abstract tasks use full phrase hints extracted deterministically from the historical abstract, not isolated words copied from the title. This heuristic improves format alignment but is not an exact reproduction of the benchmark's keyword construction. An execution audit verifies that the title and every supplied phrase reach the Answer Model intact, allowing multi-call decomposition. It does not prescribe retrieval or memory architecture. Historical generative reference strings cannot be embedded directly in candidate source; examples must be retrieved from the runtime profile.

Seed-repeat audits compare fresh requests without caching, and report output changes instead of assuming server determinism. This does not eliminate all execution noise.

## Evaluation boundary

We use user-split data and perform profile-only adaptation separately for selected test users. The evolver sees historical tasks, historical references, code and adaptation feedback. It does **not** receive the current official test input, target, or monitored test scores.

The runtime receives the current input and profile, never its reference. Test outputs are stored outside the evolution directory and are not used for candidate selection, early stopping, cohort selection, or choosing a reported best iteration. The primary comparison uses the incumbent after the predeclared 10 rounds; per-round test curves are descriptive monitoring only.

This is history-supervised test-time adaptation, not adaptation without any reference signal. Inspecting test curves while redesigning the method can still introduce researcher-level test tuning. A separate untouched cohort is needed for a final confirmatory evaluation.

The historical selection set is queried adaptively and its accept/reject outcome is visible indirectly; it is not an unbiased generalization estimate. It reduces direct fitting-set selection bias but cannot guarantee improvement on an official test task.

## Experiments

The Code Agent is **Qwen/Qwen3.8-27B in every condition**. Two frozen Answer Models are evaluated independently on the same users:

| Condition | Code Agent | Answer Model |
| --- | --- | --- |
| qwen25 | Qwen/Qwen3.8-27B | Qwen2.5-7B-Instruct |
| qwen38 | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B |

Every LLM call inside a harness uses that condition's Answer Model. PAG profile generation and CoT planning also use the condition's Answer Model. The Code Agent only evolves code. Embeddings use Qwen3-Embedding-0.6B. Existing local OpenAI-compatible services provide inference; no external commercial API is required.

Benchmarks:

- **LongLaMP:** abstract generation using the prepared user-split cohort.
- **Original LaMP:** task 2 (news categories), 3 (review ratings), 4 (news headlines), and 5 (paper titles). This uses the original category-classification LaMP-2 data, not a movie-tagging task bearing the same number.

Users across tasks are not assumed to be the same people. LaMP-1/6/7 and LaMP-QA are outside this experiment.

Every selected user must have **all four baselines and a complete RSI trajectory**:

| Method | Local implementation |
| --- | --- |
| Full Context | Concatenated history within a declared context budget |
| RAG | BM25 retrieval of two historical examples |
| PAG | Generate a natural-language user profile, then answer using it |
| CoT | Generate a history-conditioned plan, then the final answer |
| AdaPersona RSI | Independently evolved per-user Python harness |

These are reproducible local baselines, **not claims of paper-exact reproduction**. Full Context is budgeted, not an assertion that all history fits. Baselines share an 18,000-character task-plus-history safety budget across Answer Models. History is truncated when necessary; a current task over 12,000 characters raises an explicit error. Final answers allow 256 tokens for abstracts and 128 for LaMP tasks. Evolved programs may vary context and multi-call strategies within the runtime contract (8 QA calls/task). Thus the comparison is not compute-matched: report calls/runtime alongside quality and add matched-budget ablations before causal conclusions.

The initial suite uses **2 users per task**, selected in source-file order before scoring: 5 tasks × 2 Answer Models = 10 configurations. This is an operational pilot, not enough for population-level claims. Increase `--users` for larger runs.

Official tasks with missing abstract placeholders remain in the cohort. Reports annotate missing inputs and show the nonmissing subset separately; they do not remove users after looking at scores. Missing historical inputs are ineligible for constructing historical fitting/selection tasks. Users with fewer than three usable historical examples produce an explicit preparation error, not a silent test-based replacement.

## Metrics and determinism

- Text generation: report ROUGE-1, ROUGE-2, ROUGE-L, BLEU and METEOR. Optimize `0.25*ROUGE-1 + 0.35*ROUGE-L + 0.20*BLEU + 0.20*METEOR`.
- LaMP-2: optimize exact category accuracy. Report cohort-level macro-F1 separately. `label_f1` is label-string overlap for diagnostics, **not** classification F1.
- LaMP-3: optimize `1 - MAE/4`; separately report raw MAE/RMSE, accuracy and invalid outputs. Only an integer 1–5 is accepted; invalid outputs receive maximum error 4.

Quality/objective scores are displayed on a 0–100 scale. Raw MAE/RMSE are lower-is-better and remain in rating units. Do not combine different tasks into one purported benchmark score.

QA requests use temperature=0, top_p=1, top_k=1, seed=0, min_tokens=1 and thinking disabled. The one-token minimum prevents immediate EOS on tasks requiring nonempty answers; it applies equally to baselines, adaptation and test evaluation. The evolver uses temperature=0.7, top_p=0.9, without an explicit output-token cap or minimum; serving context remains a hard limit. Request settings alone do not prove server determinism. Confirmations and code hashes support auditing repeated executions.

## Run

Use Python 3.11 with numpy, scipy, scikit-learn, networkx, and:

```bash
pip install -r requirements-evolution.txt
python -m unittest discover -s tests -q
```

METEOR needs NLTK WordNet under `data/nltk_data`; metric preflight checks availability. Model serving is separate. Disposable processes provide defense in depth, not a hardened hostile-code OS sandbox. Use restricted accounts and network access.

Prepare LongLaMP with `scripts/prepare_longlamp_abstract_user_pilot.py` and `scripts/prepare_profile_only_tta.py` (see `--help`). Prepare original LaMP:

```bash
python scripts/lamp_tasks.py --source-root data/benchmarks/LaMP/user \
  --output data/experiments/lamp_user_rsi/lamp_2 --task 2 --adaptation-items 8
# Repeat for tasks 3, 4 and 5.
```

The suite reads `test.jsonl` (including each user's historical profile) under:

```text
data/experiments/longlamp_abstract_user_rsi_tta/
data/experiments/lamp_user_rsi/lamp_2/
data/experiments/lamp_user_rsi/lamp_3/
data/experiments/lamp_user_rsi/lamp_4/
data/experiments/lamp_user_rsi/lamp_5/
```

Set service addresses/model IDs in `scripts/paired_suite.py` for your deployment. Checked-in defaults are `gpu02:18012` (Qwen3.8), `gpu01:8000` (Qwen2.5), and `gpu01:18013` (embedding).

```bash
export NO_PROXY=localhost,127.0.0.1,gpu01,gpu02
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m scripts.paired_suite \
  --output-dir data/experiments/paired_main \
  --users 2 --workers 4 --iterations 10 --branches 3
```

The suite fixes a cohort manifest, rebuilds disjoint `profile_adaptation.jsonl` and `profile_selection.jsonl` using only the profiles, runs all baselines for a configuration, starts its RSI plus per-round monitor, and verifies per-user completeness before marking it complete. Four configurations run concurrently; others queue. Failures are explicitly marked. Rerun with the same directory/protocol to resume; source or configuration changes require a fresh directory. Direct evolution launches require `--selection`; use `--seed-pool` to enable the multiple starting programs.

To run only RSI, skipping the separate official-test baseline benchmark:

```bash
python -u -m scripts.paired_suite \
  --output-dir data/experiments/rsi_only_history_validation \
  --users 2 --workers 10 --user-workers 2 \
  --agent-concurrency 6 --qa-concurrency 8 \
  --iterations 10 --branches 3 --rsi-only
```

This starts up to 20 user searches across 10 configurations. Each configuration
permits up to 6 concurrent Code Agent requests and 8 Answer Model requests;
actual utilization depends on the current search phase and serving capacity.
Independent test monitors add a small separate inference load. The five
historical seed evaluations remain part of RSI initialization; `--rsi-only`
does not bypass historical selection or per-round official-test monitoring.

Artifacts under the output directory:

```text
suite.json       experiment matrix and budgets
cohorts/         paired user manifests and data hashes
baseline/        baseline code, predictions and summaries
rsi/             code, operation choices, failure banks and checkpoints
test_monitor/    seed and per-round official-test results
status/          queued/running/complete/failed per configuration
logs/            runner and monitor logs
completion.json  final completeness result
```

Datasets, weights, credentials and generated user artifacts are not uploaded to GitHub. Keep experiment outputs separate from version-controlled method code.
