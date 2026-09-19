"""Bounded cache with per-text single-flight; no cache lock during HTTP."""
from concurrent.futures import Future
from collections import OrderedDict
import json
import math
import threading
import time
import urllib.request


class EmbeddingClient:
    def __init__(self, base_url, model='Qwen3-Embedding-0.6B', timeout=120, cache_size=10000,
                 truncate_prompt_tokens=None):
        self.base_url, self.model = base_url.rstrip('/'), model
        self.timeout, self.cache_size = timeout, cache_size
        self.cache, self.pending = OrderedDict(), {}
        self.lock = threading.Lock()
        self.dimension = None
        self.truncate_prompt_tokens = truncate_prompt_tokens

    def embed(self, texts, *, instruction=None):
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError('embed expects list[str]')
        inputs = [f'Instruct: {instruction}\nQuery:{t}' if instruction else t for t in texts]
        owned, values, waiting = [], {}, {}
        with self.lock:
            for text in dict.fromkeys(inputs):
                if text in self.cache:
                    values[text] = self.cache[text]
                    self.cache.move_to_end(text)
                else:
                    if text not in self.pending:
                        self.pending[text] = Future()
                        owned.append(text)
                    waiting[text] = self.pending[text]
        try:
            for start in range(0, len(owned), 32):
                batch = owned[start:start+32]
                payload = {'model': self.model, 'input': batch}
                if self.truncate_prompt_tokens is not None:
                    payload['truncate_prompt_tokens'] = self.truncate_prompt_tokens
                request = urllib.request.Request(self.base_url + '/embeddings',
                    data=json.dumps(payload).encode(),
                    headers={'Content-Type': 'application/json'})
                for attempt in range(2):
                    try:
                        with urllib.request.urlopen(request, timeout=self.timeout) as response:
                            data = json.load(response)['data']
                        break
                    except OSError:
                        if attempt:
                            raise
                        time.sleep(.5)
                data = sorted(data, key=lambda item: item['index'])
                if [item['index'] for item in data] != list(range(len(batch))):
                    raise ValueError('embedding response has missing or duplicate indices')
                vectors = [tuple(float(x) for x in item['embedding']) for item in data]
                if any(not v or not all(math.isfinite(x) for x in v) for v in vectors):
                    raise ValueError('invalid embedding vector')
                with self.lock:
                    dimensions = {len(v) for v in vectors}
                    if len(dimensions) != 1 or (self.dimension is not None and dimensions != {self.dimension}):
                        raise ValueError('inconsistent embedding dimension')
                    self.dimension = len(vectors[0])
                    for text, vector in zip(batch, vectors):
                        self.cache[text] = vector
                        self.pending.pop(text).set_result(vector)
                    while len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
        except BaseException as error:
            with self.lock:
                for text in owned:
                    future = self.pending.pop(text, None)
                    if future is not None:
                        future.set_exception(error)
            raise
        for text, future in waiting.items():
            values[text] = future.result(timeout=self.timeout * 2 + 10)
        return [list(values[t]) for t in inputs]
