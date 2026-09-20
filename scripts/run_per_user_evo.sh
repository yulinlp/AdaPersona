#!/usr/bin/env bash
set -euo pipefail
cd /share/home/sunmeng/ylhu/AdaPersona
export NO_PROXY=127.0.0.1,localhost,gpu01,gpu02
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TRAIN=${TRAIN:-data/experiments/longlamp_abstract_user_rsi_tta/profile_adaptation.jsonl}
SELECTION=${SELECTION:?Set SELECTION to disjoint historical selection data, or use scripts.paired_suite}
OUTPUT_DIR=${OUTPUT_DIR:-data/experiments/longlamp_abstract_user_rsi_tta/runs/profile_only_v5_i10_b3_adaptive}
AGENT_URL=${AGENT_URL:-http://gpu02:18012/v1}
AGENT_MODEL=${AGENT_MODEL:-Qwen/Qwen3.8-27B}
QA_URL=${QA_URL:?Set QA_URL explicitly, or use python -m scripts.paired_suite for both answer models}
QA_MODEL=${QA_MODEL:?Set QA_MODEL explicitly}
exec /share/home/sunmeng/miniconda3/envs/vllm/bin/python -u scripts/evolve_per_user.py \
  --train "$TRAIN" --selection "$SELECTION" --seed-pool --output-dir "$OUTPUT_DIR" \
  --iterations 10 --branches 3 \
  --search-strategy archive_beam --beam-width 4 --archive-size 64 --island-count 4 \
  --agent-url "$AGENT_URL" --agent-model "$AGENT_MODEL" \
  --qa-url "$QA_URL" --qa-model "$QA_MODEL" \
  --user-workers 4 --agent-concurrency 8 "$@"
