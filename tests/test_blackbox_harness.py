import unittest

from scripts.blackbox_harness import (
    CompiledHarness,
    HarnessValidationError,
    SEED_HARNESS_CODE,
    runtime_row,
    validate_harness_code,
)
from scripts.code_evolution import (
    FailureBank,
    build_failure_entries,
    parse_proposal,
    summarize_rows,
)


class FakeQA:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "We propose a method for the current task."

    def generate_many(self, prompts, **kwargs):
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class BlackboxHarnessTest(unittest.TestCase):
    def test_seed_has_only_one_runtime_entrypoint(self):
        harness = CompiledHarness(SEED_HARNESS_CODE)
        row = {
            "user_id": "u",
            "sample_id": "s",
            "input": 'Generate an abstract for the title "Graph Networks".',
            "profile": [{"id": "p", "title": "Graph learning", "abstract": "We propose graph methods.", "year": 2020}],
        }
        output = harness.run(row, FakeQA())
        self.assertTrue(output)

    def test_runtime_contract_does_not_expose_target(self):
        harness = CompiledHarness("def run(row, qa):\n    return str('target' in row)")
        self.assertEqual(harness.run(runtime_row({"target": "secret"}), FakeQA()), "False")

    def test_unsafe_code_is_rejected(self):
        code = "import os\n\ndef run(row, qa):\n    return 'x'\n"
        self.assertTrue(validate_harness_code(code))
        with self.assertRaises(HarnessValidationError):
            CompiledHarness(code)

    def test_failure_bank_keeps_trace_without_target_text(self):
        parent = [{
            "user_id": "u",
            "sample_id": "s",
            "input": 'Generate an abstract for the title "Graph Networks".',
            "profile": [],
            "target": "SECRET GOLD ANSWER",
            "prediction": "Graph networks improve prediction.",
            "qa_trace": [{"prompt": "task", "response": "draft"}],
        }]
        child = [dict(parent[0], prediction="", error="RuntimeError: empty", qa_calls=1)]
        entries = build_failure_entries(parent, child, iteration=1, operation="micro_repair", candidate_id="c")
        self.assertTrue(entries)
        self.assertNotIn("target", entries[0])
        self.assertNotIn("SECRET GOLD ANSWER", str(entries[0]))
        self.assertIn("qa_trace", entries[0])
        self.assertGreaterEqual(summarize_rows(child)["errors"], 1)

    def test_failure_bank_round_trip(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            bank = FailureBank(Path(directory) / "failure_bank.jsonl")
            bank.add([{"failure_types": ["low_rougeL"], "sample_id": "s"}])
            loaded = FailureBank.load(bank.path)
            self.assertEqual(loaded.recent(1)[0]["sample_id"], "s")

    def test_code_only_proposal_is_accepted(self):
        parent = "def run(row, qa):\n    return 'parent'\n"
        proposal, error = parse_proposal(
            "Here is the repaired source:\n```python\ndef run(row, qa):\n    return 'child'\n```\nDone.",
            "micro_repair",
            parent,
        )
        self.assertEqual(error, "")
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["code"].splitlines()[-1], "    return 'child'")

    def test_unclosed_python_fence_is_accepted_when_source_is_complete(self):
        code = '```python\ndef run(row, qa):\n    return "ok"\n'
        self.assertEqual(CompiledHarness(code).run({}, FakeQA()), "ok")


if __name__ == "__main__":
    unittest.main()
