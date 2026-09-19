"""Versioned standard-library implementations; never silently approximate METEOR."""
from functools import lru_cache
from threading import RLock
from pathlib import Path

import nltk
nltk.data.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data' / 'nltk_data'))
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU

METRIC_PROTOCOL = 'rouge-score-0.1.2-stemmer_nltk-3.9.2-meteor_sacrebleu-2.5.1-sentence-v1'
# One source of truth for the training-time selection objective.  Values are
# kept in [0, 1]; reports expose the same score on a 0-100 scale.
OBJECTIVE_WEIGHTS = {
    'rouge1': 0.25,
    'rougeL': 0.35,
    'bleu': 0.20,
    'meteor': 0.20,
}
OBJECTIVE_PROTOCOL = 'weighted-v1:r1=0.25,rl=0.35,bleu=0.20,meteor=0.20'
_ROUGE = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
_BLEU = BLEU(effective_order=True)
_METEOR_LOCK = RLock()


def preflight():
    # Missing lexical resources are an error, not a change of scoring protocol.
    # LazyCorpusLoader mutates itself on first access. Use the same lock as
    # scoring so concurrent benchmark preflights cannot race with METEOR.
    with _METEOR_LOCK:
        wordnet.ensure_loaded()
    row_metrics({'prediction': 'a short abstract', 'target': 'a brief abstract'})
    return {'protocol': METRIC_PROTOCOL, 'objective_protocol': OBJECTIVE_PROTOCOL,
            'objective_weights': OBJECTIVE_WEIGHTS, 'rouge': 'F1, Porter stemming',
            'meteor': 'NLTK, whitespace tokens, Porter+WordNet, default parameters',
            'bleu': 'mean sentence BLEU; not corpus BLEU',
            'bleu_signature': str(_BLEU.get_signature()), 'nltk': nltk.__version__}


@lru_cache(maxsize=8192)
def _scores(prediction, reference):
    rouge = _ROUGE.score(reference, prediction)
    with _METEOR_LOCK:
        meteor = meteor_score([reference.split()], prediction.split()) if prediction and reference else 0.
    return (rouge['rouge1'].fmeasure, rouge['rouge2'].fmeasure, rouge['rougeL'].fmeasure,
            _BLEU.sentence_score(prediction, [reference]).score / 100 if prediction and reference else 0.,
            meteor, len(prediction.split()))


def row_metrics(row):
    values = _scores(str(row.get('prediction', '')), str(row.get('target', '')))
    return dict(zip(('rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'prediction_tokens'), values))


def weighted_metric_score(metrics):
    """Return the [0, 1] weighted score for one row's metric dictionary."""
    return sum(float(OBJECTIVE_WEIGHTS[name]) * float(metrics.get(name, 0.0))
               for name in OBJECTIVE_WEIGHTS)
