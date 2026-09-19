"""One sample, one disposable interpreter. JSON-only RPC; no gold or clients.

Audit restrictions are defense in depth, not a hostile-code OS sandbox.
"""
import contextlib
import json
import os
from pathlib import Path
import resource
import sys

from blackbox_harness import CompiledHarness


class Proxy:
    __slots__ = ('_reader', '_writer')

    def __init__(self, reader, writer):
        self._reader, self._writer = reader, writer

    def _request(self, method, args, kwargs):
        self._writer.write(json.dumps({'kind': 'call', 'method': method,
                                      'args': args, 'kwargs': kwargs}) + '\n')
        self._writer.flush()
        response = json.loads(self._reader.readline())
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['value']

    def generate(self, prompt, *, system='You are a helpful assistant.', max_tokens=256,
                 temperature=0.0, top_p=1.0):
        return self._request('generate', [prompt], dict(system=system, max_tokens=max_tokens,
                            temperature=temperature, top_p=top_p))

    def generate_many(self, prompts, *, system='You are a helpful assistant.', max_tokens=256,
                      temperature=0.0, top_p=1.0):
        return self._request('generate_many', [prompts], dict(system=system, max_tokens=max_tokens,
                            temperature=temperature, top_p=top_p))

    def embed(self, texts, *, instruction=None):
        return self._request('embed', [texts], {'instruction': instruction})


def restrict_io():
    roots = tuple(str(Path(p).resolve()) + os.sep for p in
                  (sys.prefix, sys.base_prefix, '/usr/lib', '/lib', '/usr/share/zoneinfo'))
    allowed_files = {'/dev/null', '/dev/urandom', '/proc/meminfo', '/proc/cpuinfo',
                     '/proc/self/stat', '/proc/self/status'}

    def audit(event, args):
        if event == 'open':
            name, mode, flags = args
            if isinstance(name, int):
                raise PermissionError('file descriptor I/O is not a harness capability')
            path = os.path.realpath(os.fspath(name))
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise PermissionError('filesystem writes are not a harness capability')
            if path not in allowed_files and not path.startswith(roots):
                raise PermissionError('only installed library resources may be read')
        if event.startswith(('socket.', 'subprocess.', 'ctypes.')) or event in {
            'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.exec',
            'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink',
            'os.chmod', 'os.truncate', 'os.chdir', 'os.putenv', 'os.unsetenv',
        }:
            raise PermissionError('external I/O/process creation is not a harness capability')
    sys.addaudithook(audit)


def main():
    reader, writer = sys.stdin, sys.stdout
    payload = json.loads(reader.readline())
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    # Import dependencies before installing I/O guards; these are trusted packages.
    # Only preload roots actually requested by the candidate.
    import ast
    import importlib
    tree = ast.parse(payload['code'])
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or '').split('.')[0])
    for root in roots & {'numpy', 'scipy', 'sklearn', 'networkx'}:
        importlib.import_module(root)
    restrict_io()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            value = CompiledHarness(payload['code']).run(payload['row'], Proxy(reader, writer))
        result = {'kind': 'result', 'value': value}
    except BaseException as error:
        result = {'kind': 'result', 'error': type(error).__name__ + ': ' + str(error)}
    writer.write(json.dumps(result) + '\n')
    writer.flush()


if __name__ == '__main__':
    main()
