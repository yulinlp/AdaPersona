"""History-only task construction and disjoint fitting/selection partitions.

This module never accepts the official current input or reference answer.
"""
from collections import Counter
import hashlib
import json
import re

PROTOCOL = 'profile-fit-selection-v1'
MISSING = {'', 'no abstract available', 'without abstract', 'abstract unavailable', 'n/a', 'none'}


def normalized(value):
    return ' '.join(re.findall(r'\w+', str(value).casefold()))


def missing_text(value):
    return normalized(value) in MISSING


def history_key(item):
    # Content identity catches duplicated examples with different dataset IDs.
    return hashlib.sha256(json.dumps({k: v for k, v in item.items() if k != 'id'},
                                     sort_keys=True).encode()).hexdigest()


def abstract_input(title, abstract):
    """Deterministic phrase hints from the historical abstract, not title words.

    These are self-supervised hints, not a claim to reproduce the benchmark's
    unpublished keyword generator. Multiword phrases remain intact.
    """
    stop = set('a an the and or but of in on at for with by from to is are was were be been '
               'this that these those we our it its as into through which using use used '
               'paper present propose show study results also can has have will'.split())
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[.,;:!?()]", str(abstract))
    chunks, current = [], []
    for word in words + ['.']:
        if word.lower() in stop or not re.search(r'\w', word):
            if current:
                chunks.extend(' '.join(current[i:i+5]) for i in range(0, len(current), 5))
            current = []
        else:
            current.append(word)
    counts = Counter(normalized(p) for p in chunks)
    title_terms = set(normalized(title).split())
    ranked = sorted(enumerate(chunks), key=lambda pair: (
        -(len(title_terms & set(normalized(pair[1]).split())) + counts[normalized(pair[1])]),
        -min(len(pair[1].split()), 4), pair[0]))
    phrases, seen = [], set()
    for _, phrase in ranked:
        key = normalized(phrase)
        if key not in seen and len(key) > 2:
            seen.add(key)
            phrases.append(phrase)
        if len(phrases) == 5:
            break
    suffix = '\n'.join(f'{i}. {phrase}' for i, phrase in enumerate(phrases, 1))
    return f'Generate an abstract for the title "{title}" using the following items: {suffix}'


def build_history_partitions(user_id, profile, benchmark, task=None, fit_limit=8, selection_limit=4):
    try:
        from .lamp_tasks import history_task
    except ImportError:
        from lamp_tasks import history_task
    unique, seen = [], set()
    for item in profile:
        key = history_key(item)
        if key in seen:
            continue
        seen.add(key)
        if benchmark == 'longlamp':
            if missing_text(item.get('abstract')) or missing_text(item.get('title')):
                continue
            pair = (abstract_input(item['title'], item['abstract']), str(item['abstract']))
        else:
            field = 'abstract' if task == 5 else 'text'
            if missing_text(item.get(field)):
                continue
            pair = history_task(task, item)
            if pair is None:
                continue
        unique.append((key, item, pair))
    ranked = sorted(unique, key=lambda x: hashlib.sha256(f'{user_id}:{x[0]}'.encode()).hexdigest())
    if len(ranked) < 3:
        raise ValueError(f'{user_id}: need at least three usable unique historical examples; no test-based replacement')
    selection_count = min(selection_limit, max(1, len(ranked)//4))
    selection = ranked[:selection_count]
    fit = ranked[selection_count:selection_count+fit_limit]
    protected = {key for key, _, _ in selection}
    # No selection examples in fitting profiles, prompts, traces or failure banks.
    fit_profile = [item for item in profile if history_key(item) not in protected]
    def make(chosen, pool, split):
        return [dict(user_id=user_id, sample_id=f'history_{split}:{user_id}:{key[:16]}',
                     input=pair[0], target=pair[1], profile=[p for p in pool if history_key(p) != key],
                     source_split=f'profile_{split}', origin=PROTOCOL,
                     historical_key=key, benchmark=benchmark, task=task,
                     input_contract='preserve_task_hints' if benchmark == 'longlamp' else 'native_task')
                for key, _, pair in chosen]
    # Selection prompts also omit all selection examples, including each other.
    return make(fit, fit_profile, 'fit'), make(selection, fit_profile, 'selection'), dict(
        usable_history=len(unique), fit=len(fit), selection=len(selection),
        selection_keys=sorted(protected), protocol=PROTOCOL)


def task_hint_loss(row, traces):
    """Audit only explicit title/keyword requirements, not internal architecture."""
    if row.get('input_contract') != 'preserve_task_hints':
        return []
    text = str(row['input'])
    title = re.search(r'title\s+"([^"]+)"', text, re.I)
    hints = re.findall(r'(?:^|\n|items:\s*)\s*\d+\.\s*([^\n]+)', text, re.I)
    required = ([title.group(1)] if title else []) + hints
    prompts = [normalized(str(t.get('prompt', ''))+' '+str(t.get('system', '')))
               for t in traces if 'prompt' in t]
    # Allow multi-call decompositions, but every provided fact must reach an LLM.
    return [value for value in required if not any(normalized(value) in prompt for prompt in prompts)]
