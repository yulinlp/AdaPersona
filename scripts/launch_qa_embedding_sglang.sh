#!/usr/bin/env bash
# One authorized A100: deterministic native SGLang QA + separate vLLM embedding.
#SBATCH -J ada-qa-sgl-embed
#SBATCH -p alpha
#SBATCH -N 1
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 7-0:00:00
#SBATCH -o /share/home/sunmeng/ylhu/AdaPersona/logs/ada-qa-sgl-embed-%j.out
#SBATCH -e /share/home/sunmeng/ylhu/AdaPersona/logs/ada-qa-sgl-embed-%j.err
set -euo pipefail
cd /share/home/sunmeng/ylhu/AdaPersona
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NO_PROXY=127.0.0.1,localhost,gpu01,gpu02
export no_proxy="$NO_PROXY"
export PATH=/share/public/apps/gcc/12.2.0/bin:/share/home/sunmeng/miniconda3/envs/vllm_env/bin:$PATH
export CC=/share/public/apps/gcc/12.2.0/bin/gcc
export CXX=/share/public/apps/gcc/12.2.0/bin/g++
export LD_LIBRARY_PATH=/share/public/apps/gcc/12.2.0/lib64:${LD_LIBRARY_PATH:-}
qa_pid=''
embed_pid=''
cleanup() {
  if [[ -n "$qa_pid" ]]; then kill -TERM "$qa_pid" 2>/dev/null || true; fi
  if [[ -n "$embed_pid" ]]; then kill -TERM "$embed_pid" 2>/dev/null || true; fi
  wait || true
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# Reuse the matched CUDA-12.8 SGLang/kernel environment, without modifying it
# or mixing in the unrelated CUDA-13 overlay used by earlier diagnostics.
sglang_path=/share/home/sunmeng/ylhu/steem-adaptive-steering/runs/causal_activation_patching/local_judge_qwen38/sglang_env/lib/python3.10/site-packages
env PYTHONPATH="$sglang_path" \
 /share/home/sunmeng/miniconda3/envs/vllm_env/bin/python -m sglang.launch_server \
  --model-path /share/home/sunmeng/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --context-length 8192 --tp-size 1 \
  --mem-fraction-static 0.78 --max-running-requests 32 --max-queued-requests 256 \
  --chunked-prefill-size 4096 --max-prefill-tokens 8192 \
  --attention-backend triton --enable-deterministic-inference --random-seed 0 \
  --disable-radix-cache --disable-overlap-schedule --disable-cuda-graph \
  > "logs/qa-sglang-${SLURM_JOB_ID:-manual}.log" 2>&1 &
qa_pid=$!
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

/share/home/sunmeng/miniconda3/envs/vllm/bin/vllm serve \
  /share/home/sunmeng/models/Qwen3-Embedding-0.6B \
  --served-model-name Qwen3-Embedding-0.6B --host 0.0.0.0 --port 18013 \
  --runner pooling --dtype bfloat16 --max-model-len 2048 \
  --pooler-config '{"pooling_type":"LAST","normalize":true}' \
  --gpu-memory-utilization 0.10 --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --enforce-eager --disable-log-requests \
  > "logs/embedding-sgl-colocated-${SLURM_JOB_ID:-manual}.log" 2>&1 &
embed_pid=$!
printf 'QA_PID=%s EMBEDDING_PID=%s GPU=0 QA_BACKEND=sglang\n' "$qa_pid" "$embed_pid"
wait -n "$qa_pid" "$embed_pid"
exit 1
