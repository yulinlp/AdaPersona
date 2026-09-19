"""Read-only diagnostic: freeze one harness request and retain replay evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from embedding_client import EmbeddingClient
from evolution_metrics import row_metrics, weighted_metric_score


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--user', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    row = next(json.loads(line) for line in Path(args.test).read_text().splitlines()
               if json.loads(line)['user_id'] == args.user)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    requests = []

    class Capture:
        def __init__(self):
            self.embedding = EmbeddingClient('http://gpu01:18013/v1', truncate_prompt_tokens=2048)
        def embed(self, texts, **kwargs):
            return self.embedding.embed(texts, **kwargs)
        def generate(self, prompt, *, system, max_tokens, temperature=0, top_p=1):
            requests.append(dict(model='Qwen/Qwen3.8-27B', messages=[
                dict(role='system', content=system), dict(role='user', content=prompt)],
                max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=0,
                repetition_penalty=1.0, n=1, stream=False,
                chat_template_kwargs={'enable_thinking': False}))
            return 'capture only'

    # Use the normal sandbox interface; do not expose the test reference to code.
    from isolated_runtime import run_isolated
    from blackbox_harness import runtime_row
    code = Path(args.code).read_text()
    capture = Capture()
    for _ in range(3):
        run_isolated(code, runtime_row(row), capture, timeout=120)
    (output / 'requests.json').write_text(json.dumps(requests, ensure_ascii=False, indent=2))
    print('REQUEST_HASHES', [digest(r) for r in requests], flush=True)
    assert len(requests) == 3 and len({digest(r) for r in requests}) == 1
    payload = requests[0]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def replay(index):
        start = time.time()
        req = urllib.request.Request('http://gpu02:18012/v1/chat/completions',
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=120) as response:
            raw = json.load(response)
        prediction = raw['choices'][0]['message']['content']
        metrics = row_metrics({'prediction': prediction, 'target': row['target']})
        record = dict(index=index, seconds=time.time()-start, request_hash=digest(payload),
            raw=raw, metrics=metrics, score=100*weighted_metric_score(metrics))
        (output / f'replay_{index:02d}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print('RESULT', index, round(record['score'], 3), digest(prediction), flush=True)
        return record

    results = [replay(i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results.extend(pool.map(replay, range(4, 12)))
    (output / 'summary.json').write_text(json.dumps(dict(
        code_hash=hashlib.sha256(code.encode()).hexdigest(), row_hash=digest(row),
        scores=[r['score'] for r in results], request_hash=digest(payload)), indent=2))


if __name__ == '__main__':
    main()
