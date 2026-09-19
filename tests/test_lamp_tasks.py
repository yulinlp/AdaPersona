import json
from pathlib import Path
import tempfile
import unittest

from scripts.blackbox_harness import CompiledHarness, runtime_row
from scripts.lamp_tasks import (
    LAMP_SEED_HARNESS_CODE,
    LampTaskAdapter,
    _task2_input,
    prepare_user_test,
)


class FixedQA:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, prompt, **kwargs):
        return self.answer


class LampTaskTest(unittest.TestCase):
    def test_task_two_uses_native_category_contract(self):
        prompt = _task2_input({"title": "A title", "text": "An article"})
        self.assertIn("style & beauty", prompt)
        self.assertIn("science & technology", prompt)
        self.assertIn("healthy living", prompt)
        self.assertNotIn("other", prompt)

    def test_classification_and_rating_objectives(self):
        classification = LampTaskAdapter(2)
        metrics = classification.row_metrics({
            "prediction": "healthy living", "target": "healthy living"
        })
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(classification.weighted_score(metrics), 1.0)
        summary = classification.summarize_rows([{
            "user_id": "u", "sample_id": "s", "prediction": "healthy living",
            "target": "healthy living", "input": "classify", "profile": [],
        }])
        self.assertEqual(summary["weighted_score_100"], 100.0)

        rating = LampTaskAdapter(3)
        metrics = rating.row_metrics({"prediction": "4", "target": "5"})
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["rating_closeness"], 0.75)

    def test_prepare_native_user_test_is_profile_only_leave_one_out(self):
        source = Path("data/benchmarks/LaMP/user")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            stats = prepare_user_test(
                source_root=source, output=output, task=4,
                adaptation_items=3, max_users=2,
            )
            self.assertEqual(stats["users_in_test"], 2)
            self.assertFalse(stats["official_test_target_used_during_adaptation"])
            test_rows = [json.loads(line) for line in
                         (output / "test.jsonl").read_text().splitlines()]
            adaptation_rows = [json.loads(line) for line in
                               (output / "profile_adaptation.jsonl").read_text().splitlines()]
            self.assertEqual(len(test_rows), 2)
            self.assertEqual(len(adaptation_rows), 6)
            for row in adaptation_rows:
                profile_ids = {str(item.get("id")) for item in row["profile"]}
                self.assertNotIn(row["held_out_profile_item"], profile_ids)
                self.assertTrue(row["target"])
            self.assertTrue(all(row["target"] for row in test_rows))

    def test_seed_keeps_fixed_harness_interface_and_task_profile(self):
        harness = CompiledHarness(LAMP_SEED_HARNESS_CODE)
        row = runtime_row({
            "user_id": "lamp2:1", "sample_id": "s", "input": "classify",
            "target": "secret", "profile": [{"text": "food", "title": "Food",
                                                  "category": "food & drink", "id": "1"}],
        })
        output = harness.run(row, FixedQA("food & drink"))
        self.assertEqual(output, "food & drink")
        self.assertNotIn("target", row)


if __name__ == "__main__":
    unittest.main()
