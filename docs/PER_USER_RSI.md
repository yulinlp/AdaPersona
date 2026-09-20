# Per-user history-validated Python harness evolution

The canonical paired launcher is `python -m scripts.paired_suite`; see the
[README](../README.md) for the complete experiment matrix and service setup.

## Data boundary

`scripts/profile_protocol.py` receives only a user ID and historical profile.
It constructs up to 8 fitting and 4 selection tasks, with smaller partitions
for short histories. Selection examples are absent from fitting inputs,
profiles, references and failure banks. Each historical task omits its own
source item. Missing historical inputs are ineligible for task construction.

LongLaMP historical hints are full phrases extracted from historical abstracts.
They are an explicitly heuristic approximation to task format, not an exact
replication of official keyword construction. Original LaMP tasks 2/3/4/5 use
native historical input/target pairs.

The official current test task is not used to build these partitions. Its
reference is only available to the independent monitor. Missing official
inputs remain in the cohort and are annotated rather than silently excluded.

## Search and selection

The Code Agent is Qwen3.8-27B; frozen Answer Models are separately Qwen2.5-7B
and Qwen3.8-27B. Each user has a separate executable `run(row, qa) -> str`.

The retrieval seed plus Full Context, RAG, PAG and CoT are compared on history.
The seed is selected by historical selection score (fitting score breaks ties).
Valid alternatives remain in the archive. Seed-repeat audits use fresh requests
and report prediction differences rather than claiming perfect determinism.

Each of 10 rounds chooses either macro strategy exploration or micro repair,
then proposes at most 3 candidates total. Operations may repeat consecutively.
Archive-beam defaults to one incumbent slot and remaining slots for historical
alternatives; this policy is configurable. Candidate internals remain free-form.

Candidate ranking prioritizes historical selection score. Acceptance requires
zero errors, strict fitting improvement, and no selection regression, including
fresh candidate/parent confirmations. Selection details never enter the Code
Agent prompt or failure bank. Repeated accept/reject decisions still use the
selection set adaptively: it is not an untouched generalization test.

Full current titles and keyword phrases must reach the Answer Model, possibly
across multiple calls. Direct historical generative-reference literals are
rejected; category/rating vocabularies remain allowed. These guards cannot prove
absence of semantic memorization or enforce preservation of all meaning.

## Monitoring and recovery

`scripts/monitor_lamp_test.py` supports both benchmarks and scores seed plus
every committed round, writing outside the evolution directory. Never select
a winning iteration using these official test scores. Report the predeclared
last-round incumbent and preserve incomplete/error cases explicitly.

Checkpointed operation choices have bounded formatting-repair retries. Round
completion embeds its monitor event in the atomic state, allowing recovery
after interruption. Configuration/data/source changes require a new run
directory. Existing runs are not silently migrated to a different protocol.

Direct launches require `--train`, `--selection` and matching user cohorts.
Use `--seed-pool` to enable the multiple starting programs. Historical selection
files require the provenance/key fields emitted by the paired suite; legacy
preparation files without isolation guarantees are intentionally rejected.
