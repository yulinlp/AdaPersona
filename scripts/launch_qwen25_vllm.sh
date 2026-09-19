#!/usr/bin/env bash
#SBATCH -J ada-qwen25
#SBATCH -p alpha
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 7-0:00:00
#SBATCH --signal=B:USR1@300
#SBATCH -o /share/home/sunmeng/ylhu/AdaPersona/logs/ada-qwen25-%j.out
#SBATCH -e /share/home/sunmeng/ylhu/AdaPersona/logs/ada-qwen25-%j.err

set -euo pipefail

exec /share/home/sunmeng/miniconda3/envs/vllm/bin/vllm serve \
  /share/home/sunmeng/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests
