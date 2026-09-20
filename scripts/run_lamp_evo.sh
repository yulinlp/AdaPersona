#!/usr/bin/env bash
set -euo pipefail

cd /share/home/sunmeng/ylhu/AdaPersona
export NO_PROXY=127.0.0.1,localhost,gpu01,gpu02
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

TASK=${TASK:-4}
DATA_ROOT=${DATA_ROOT:-data/experiments/lamp_user_rsi}
TRAIN=${TRAIN:-$DATA_ROOT/lamp_${TASK}/profile_adaptation.jsonl}
SELECTION=${SELECTION:?Set SELECTION to disjoint historical selection data, or use scripts.paired_suite}
OUTPUT_DIR=${OUTPUT_DIR:-$DATA_ROOT/lamp_${TASK}/runs/qwen38_i10_b3_adaptive}
AGENT_URL=${AGENT_URL:-http://gpu02:18012/v1}
AGENT_MODEL=${AGENT_MODEL:-Qwen/Qwen3.8-27B}
QA_URL=${QA_URL:?Set QA_URL explicitly, or use python -m scripts.paired_suite for both answer models}
QA_MODEL=${QA_MODEL:?Set QA_MODEL explicitly}
ITERATIONS=${ITERATIONS:-10}
BRANCHES=${BRANCHES:-3}
USER_WORKERS=${USER_WORKERS:-4}
AGENT_CONCURRENCY=${AGENT_CONCURRENCY:-8}

extra_args=()
if [[ -n "${MAX_USERS:-}" ]]; then
  extra_args+=(--max-users "$MAX_USERS")
fi

exec /share/home/sunmeng/miniconda3/envs/vllm/bin/python -u scripts/evolve_per_user.py \
  --benchmark lamp --task "$TASK" \
  --train "$TRAIN" --selection "$SELECTION" --seed-pool --output-dir "$OUTPUT_DIR" \
  --iterations "$ITERATIONS" --branches "$BRANCHES" \
  --search-strategy archive_beam --beam-width 4 --archive-size 64 --island-count 4 \
  --agent-url "$AGENT_URL" --agent-model "$AGENT_MODEL" \
  --qa-url "$QA_URL" --qa-model "$QA_MODEL" \
  --user-workers "$USER_WORKERS" --agent-concurrency "$AGENT_CONCURRENCY" \
  "${extra_args[@]}" "$@"
