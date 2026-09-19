import io
import json
import unittest
from unittest.mock import patch
from scripts.embedding_client import EmbeddingClient
from scripts.blackbox_harness import CompiledHarness, VLLMQA

class EmbeddingToolsTest(unittest.TestCase):
    def test_vllm_truncation_parameter_is_explicit(self):
        def response(request, timeout):
            payload = json.loads(request.data)
            self.assertEqual(payload['truncate_prompt_tokens'],2048)
            return io.BytesIO(json.dumps({'data':[{'index':0,'embedding':[1.,0.]}]}).encode())
        with patch('urllib.request.urlopen',side_effect=response):
            client=EmbeddingClient('http://localhost/v1',truncate_prompt_tokens=2048)
            self.assertEqual(client.embed(['long text']),[[1.,0.]])

    def test_order_duplicates_instruction_and_cache_isolation(self):
        def response(request, timeout):
            inputs = json.loads(request.data)['input']
            return io.BytesIO(json.dumps({'data': [
                {'index': i, 'embedding': [float(i), 1.]}
                for i in reversed(range(len(inputs)))]}).encode())
        client = EmbeddingClient('http://localhost/v1')
        with patch('urllib.request.urlopen', side_effect=response) as call:
            first = client.embed(['a', 'b', 'a'])
            self.assertEqual(first, [[0., 1.], [1., 1.], [0., 1.]])
            first[0][0] = 99
            self.assertEqual(client.embed(['a']), [[0., 1.]])
            self.assertEqual(call.call_count, 1)
            client.embed(['a'], instruction='retrieve')
            self.assertEqual(call.call_count, 2)
            self.assertEqual(json.loads(call.call_args.args[0].data)['input'],
                             ['Instruct: retrieve\nQuery:a'])

    def test_runtime_tools(self):
        class FakeEmbedding:
            def embed(self, texts, instruction=None):
                return [[1., 0.], [0., 1.]]
        qa = VLLMQA(None, embedding_client=FakeEmbedding())
        harness = CompiledHarness('''import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
def run(row, qa):
    vectors = np.asarray(qa.embed(['a', 'b']))
    g = nx.Graph()
    g.add_edge('a', 'b')
    scores = cosine_similarity(csr_matrix(vectors))
    return str((nx.shortest_path(g, 'a', 'b'), float(scores[0, 0])))
''')
        self.assertEqual(harness.run({}, qa), "(['a', 'b'], 1.0)")
        self.assertEqual(qa.calls, 0)
        self.assertEqual(qa.trace[0]['dimension'], 2)

if __name__ == '__main__':
    unittest.main()
