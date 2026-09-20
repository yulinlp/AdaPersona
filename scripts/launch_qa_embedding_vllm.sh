#!/usr/bin/env bash
#SBATCH -J ada-qa-embed
#SBATCH -p alpha
#SBATCH -N 1
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 7-0:00:00
#SBATCH -o /share/home/sunmeng/ylhu/AdaPersona/logs/ada-qa-embed-%j.out
#SBATCH -e /share/home/sunmeng/ylhu/AdaPersona/logs/ada-qa-embed-%j.err
set -euo pipefail
cd /share/home/sunmeng/ylhu/AdaPersona
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost,gpu01,gpu02
export no_proxy="$NO_PROXY"
export PYTHONHASHSEED=0
# A100 diagnostic/reproducible serving mode. Queue concurrent HTTP requests,
# but run one QA sequence at a time with no prefix-cache-dependent prefill.
# Do not claim batch invariance from temperature/seed settings alone.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
serve_bin=/share/home/sunmeng/miniconda3/envs/vllm/bin/vllm
qa_pid=''
embed_pid=''
cleanup() {
  if [[ -n "$qa_pid" ]]; then kill -TERM "$qa_pid" 2>/dev/null || true; fi
  if [[ -n "$embed_pid" ]]; then kill -TERM "$embed_pid" 2>/dev/null || true; fi
  wait || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT

"$serve_bin" serve /share/home/sunmeng/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.78 \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --enforce-eager \
  --seed 0 --disable-log-requests \
  > "logs/qa-colocated-${SLURM_JOB_ID:-manual}.log" 2>&1 &
qa_pid=$!

# Profile QA memory first, then start the second instance against known free space.
qa_ready=false
for attempt in $(seq 1 180); do
  kill -0 "$qa_pid" || exit 1
  if curl --noproxy '*' -sf --max-time 2 http://127.0.0.1:8000/health >/dev/null; then
    qa_ready=true
    break
  fi
  sleep 5
done
[[ "$qa_ready" == true ]] || exit 1

"$serve_bin" serve /share/home/sunmeng/models/Qwen3-Embedding-0.6B \
  --served-model-name Qwen3-Embedding-0.6B --host 0.0.0.0 --port 18013 \
  --runner pooling --dtype bfloat16 --max-model-len 2048 \
  --pooler-config '{"pooling_type":"LAST","normalize":true}' \
  --gpu-memory-utilization 0.10 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --enforce-eager --disable-log-requests \
  > "logs/embedding-vllm-${SLURM_JOB_ID:-manual}.log" 2>&1 &
embed_pid=$!
printf 'QA_PID=%s EMBEDDING_PID=%s GPU=0\n' "$qa_pid" "$embed_pid"
# Failure of either service terminates the allocation rather than hiding a half-service.
wait -n "$qa_pid" "$embed_pid"
exit 1
