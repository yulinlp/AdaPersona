import tempfile
import unittest
from pathlib import Path

from scripts.search_policy import dominates, metric_keys, pareto_front, select_parent_branches


def summary(r1, rl, bleu, meteor):
    return {
        'errors': 0,
        'mean_rouge1': r1,
        'mean_rougeL': rl,
        'mean_bleu': bleu,
        'mean_meteor': meteor,
        'weighted_score': .25*r1 + .35*rl + .20*bleu + .20*meteor,
    }


class SearchPolicyTest(unittest.TestCase):
    def test_task_specific_objective_dimensions_are_visible(self):
        left = {
            'objective_weights': {'accuracy': .75, 'label_f1': .25},
            'mean_accuracy': .9, 'mean_label_f1': .3,
        }
        right = {
            'objective_weights': {'accuracy': .75, 'label_f1': .25},
            'mean_accuracy': .8, 'mean_label_f1': .2,
        }
        self.assertEqual(metric_keys(left), ('mean_accuracy', 'mean_label_f1'))
        self.assertTrue(dominates(left, right))

    def test_pareto_front_keeps_complementary_candidates(self):
        strong_rl = {'summary': summary(.3, .9, .3, .3)}
        strong_bleu = {'summary': summary(.3, .3, .9, .3)}
        dominated = {'summary': summary(.2, .2, .2, .2)}
        self.assertTrue(dominates(strong_rl['summary'], dominated['summary']))
        self.assertEqual(set(map(id, pareto_front([strong_rl, strong_bleu, dominated]))),
                         {id(strong_rl), id(strong_bleu)})

    def test_archive_selection_expands_multiple_lineages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / 'seed.py'
            seed.write_text('def run(row, qa):\n    return qa.generate(row["input"])\n')
            pred = root / 'seed.jsonl'
            pred.write_text('')
            incumbent = {
                'candidate_id': 'seed', 'code_path': str(seed),
                'predictions_path': str(pred), 'summary': summary(.5, .5, .5, .5),
            }
            archive = []
            for i, text in enumerate([
                'def run(row, qa):\n    return qa.generate("A " + row["input"])\n',
                'def run(row, qa):\n    return qa.generate("B " + row["input"])\n',
                'def run(row, qa):\n    if row["profile"]:\n        return qa.generate(row["input"], max_tokens=128)\n    return qa.generate(row["input"])\n',
            ], 1):
                path = root / f'candidate_{i}.py'
                path.write_text(text)
                archive.append({
                    'candidate_id': f'c{i}', 'code_path': str(path),
                    'predictions_path': str(pred), 'errors': 0,
                    'mean_rouge1': .4 + .03*i, 'mean_rougeL': .45 + .01*i,
                    'mean_bleu': .4 + .04*i, 'mean_meteor': .45,
                    'weighted_score': .25*(.4+.03*i)+.35*(.45+.01*i)+.20*(.4+.04*i)+.20*.45,
                })
            groups = select_parent_branches(
                incumbent=incumbent, archive=archive, branch_count=4,
                beam_width=4, archive_size=64, island_count=4,
                iteration=1, operation='macro_strategy')
            self.assertEqual(sum(count for _, count in groups), 4)
            self.assertGreaterEqual(len(groups), 2)
            incumbent_slots = sum(count for node, count in groups
                                  if node['code_path'] == str(seed))
            self.assertEqual(incumbent_slots, 2)

    def test_archive_search_reserves_incumbent_slots_when_archive_is_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / 'seed.py'
            seed.write_text('def run(row, qa):\n    return qa.generate(row["input"])\n')
            pred = root / 'seed.jsonl'
            pred.write_text('')
            incumbent = {
                'candidate_id': 'seed', 'code_path': str(seed),
                'predictions_path': str(pred), 'summary': summary(.8, .8, .8, .8),
            }
            archive = []
            for i in range(8):
                path = root / f'candidate_{i}.py'
                path.write_text(
                    'def run(row, qa):\n'
                    f'    return qa.generate({i!r} + row["input"])\n'
                )
                archive.append({
                    'candidate_id': f'c{i}', 'code_path': str(path),
                    'predictions_path': str(pred), 'errors': 0,
                    'mean_rouge1': .3 + .01*i, 'mean_rougeL': .3 + .01*i,
                    'mean_bleu': .3 + .01*i, 'mean_meteor': .3 + .01*i,
                    'weighted_score': .3 + .01*i,
                })
            groups = select_parent_branches(
                incumbent=incumbent, archive=archive, branch_count=4,
                beam_width=4, archive_size=64, island_count=4,
                iteration=5, operation='micro_repair')
            self.assertEqual(sum(count for _, count in groups), 4)
            self.assertEqual(sum(count for node, count in groups
                                 if node['code_path'] == str(seed)), 2)
            self.assertEqual(len(groups), 3)


if __name__ == '__main__':
    unittest.main()
