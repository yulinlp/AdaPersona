"""Local frozen text embeddings; CPU by default, bounded batch inference."""
import argparse
import threading
from collections import OrderedDict

def main():
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=18013)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    cache = OrderedDict()
    lock = threading.Lock()
    app = FastAPI()
    class Request(BaseModel):
        model: str = 'Qwen3-Embedding-0.6B'
        input: list[str]

    @app.get('/health')
    def health():
        return {'model': args.model, 'device': args.device, 'max_length': args.max_length,
                'cached_texts': len(cache)}

    @app.post('/v1/embeddings')
    def embeddings(body: Request):
        if body.model != 'Qwen3-Embedding-0.6B':
            raise HTTPException(400, 'Unknown embedding model')
        with lock, torch.inference_mode():
            missing = list(dict.fromkeys(t for t in body.input if t not in cache))
            computed = {}
            for start in range(0, len(missing), args.batch_size):
                texts = missing[start:start + args.batch_size]
                batch = tokenizer(texts, padding=True, truncation=True,
                                  max_length=args.max_length, return_tensors='pt').to(args.device)
                vectors = model(**batch).last_hidden_state[:, -1]
                vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1).cpu().tolist()
                computed.update(zip(texts, vectors))
            result = [computed[t] if t in computed else cache[t] for t in body.input]
            cache.update(computed)
            for text in body.input:
                cache.move_to_end(text)
            while len(cache) > 50000:
                cache.popitem(last=False)
        return {'object': 'list', 'model': body.model, 'data': [
            {'object': 'embedding', 'index': i, 'embedding': vector}
            for i, vector in enumerate(result)]}

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
