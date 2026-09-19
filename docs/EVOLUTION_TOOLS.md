# V5 runtime tools

The V5 runtime contract is `run(row, qa) -> str`.

The QA facade provides:

- `qa.generate(prompt, system=..., max_tokens=..., temperature=..., top_p=...)`
- `qa.generate_many(prompts, system=..., max_tokens=..., temperature=..., top_p=...)`
- `qa.embed(texts, instruction=None)`

All runtime calls go through the selected frozen QA endpoint. The default is
Qwen2.5-7B-Instruct at `http://gpu01:8000/v1`; Qwen3.8 at
`http://gpu02:18012/v1` is also supported as a QA model. The outer evolver is
configured independently with `--agent-url` and `--agent-model`.

The default embedding endpoint is Qwen3-Embedding-0.6B at
`http://gpu01:18013/v1`. Embeddings are normalized 1024-dimensional vectors,
with 2048-token input truncation, a 4096-text/2,000,000-character per-sample
budget, and a bounded client cache. Retrieval and embeddings are optional
runtime strategies; no fixed module decomposition is imposed.

Start the colocated default services with:

```bash
sbatch scripts/launch_qa_embedding_vllm.sh
```

Run V5 with:

```bash
bash scripts/run_per_user_evo.sh
```

Override `TRAIN`, `OUTPUT_DIR`, `AGENT_URL`, `AGENT_MODEL`, `QA_URL`, and
`QA_MODEL` as environment variables, or pass additional CLI flags after the
wrapper. The wrapper defaults to 10 iterations, four branches, archive-beam
search, and four user workers.

Install metric dependencies with `requirements-evolution.txt`; the local
dataset, model weights, service logs, and run artifacts are deliberately not
versioned.
