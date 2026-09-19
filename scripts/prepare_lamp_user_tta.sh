#!/usr/bin/env bash
set -euo pipefail

cd /share/home/sunmeng/ylhu/AdaPersona
export NO_PROXY=127.0.0.1,localhost,gpu01,gpu02
export no_proxy="$NO_PROXY"

SOURCE_ROOT=${SOURCE_ROOT:-data/benchmarks/LaMP/user}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/experiments/lamp_user_rsi}
ADAPTATION_ITEMS=${ADAPTATION_ITEMS:-8}
MAX_USERS=${MAX_USERS:-}

extra_args=()
if [[ -n "$MAX_USERS" ]]; then
  extra_args+=(--max-users "$MAX_USERS")
fi

for task in 2 3 4 5; do
  /share/home/sunmeng/miniconda3/envs/vllm/bin/python scripts/lamp_tasks.py \
    --source-root "$SOURCE_ROOT" \
    --output "$OUTPUT_ROOT/lamp_${task}" \
    --task "$task" \
    --adaptation-items "$ADAPTATION_ITEMS" \
    "${extra_args[@]}"
done
