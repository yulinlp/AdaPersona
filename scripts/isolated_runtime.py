"""Parent-side broker: only explicitly exposed frozen-model capabilities."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

# Requests can outlive a killed candidate, but never hold its process alive.
# HTTP clients have their own finite timeouts; global QA limits remain enforced.
_RPC = ThreadPoolExecutor(max_workers=32, thread_name_prefix='harness-rpc')


def run_isolated(code, row, qa, timeout=300):
    worker = Path(__file__).with_name('harness_worker.py').resolve()
    env = {k: v for k, v in os.environ.items() if k in
           {'PATH', 'LANG', 'LC_ALL', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'}}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='personalized-harness-') as directory:
        process = subprocess.Popen([sys.executable, '-B', str(worker)], cwd=directory, env=env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, text=True, bufsize=1)
        pending = None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        try:
            process.stdin.write(json.dumps({'code': code, 'row': row}) + '\n')
            process.stdin.flush()
            while time.monotonic() < deadline:
                if pending is not None:
                    if not pending.done():
                        time.sleep(.01)
                        continue
                    try:
                        response = {'value': pending.result()}
                    except Exception as error:
                        response = {'error': type(error).__name__ + ': ' + str(error)}
                        qa.trace.append({'kind': 'rpc_error', **response})
                    process.stdin.write(json.dumps(response) + '\n')
                    process.stdin.flush()
                    pending = None
                if not selector.select(timeout=min(.05, max(0, deadline-time.monotonic()))):
                    continue
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('isolated harness exited without a result')
                message = json.loads(line)
                if message.get('kind') == 'result':
                    if message.get('error'):
                        raise RuntimeError(message['error'])
                    return message['value']
                method = message.get('method')
                if message.get('kind') != 'call' or method not in ('generate', 'generate_many', 'embed'):
                    raise RuntimeError('invalid harness RPC capability')
                pending = _RPC.submit(getattr(qa, method), *message['args'], **message['kwargs'])
            raise TimeoutError(f'harness exceeded {timeout}s wall-clock limit')
        finally:
            if pending is not None:
                pending.cancel()
            selector.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
