"""Compare fresh executions without replaying or caching model responses."""


def compare_executions(first, second):
    def keyed(rows):
        result = {(str(r.get('user_id', '')), str(r['sample_id'])): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate execution audit sample IDs')
        return result

    a, b = keyed(first), keyed(second)
    changes = []
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            changes.append(dict(user_id=key[0], sample_id=key[1], cause='sample_set_changed'))
            continue
        left, right = a[key], b[key]
        cause = None
        lt, rt = left.get('qa_trace', []), right.get('qa_trace', [])
        # Attribute the FIRST divergence, not downstream prompts that merely
        # incorporate an earlier changed model response.
        for x, y in zip(lt, rt):
            if {k:v for k,v in x.items() if k != 'response'} != {k:v for k,v in y.items() if k != 'response'}:
                cause = 'request_or_runtime_changed'
                break
            if x.get('response') != y.get('response'):
                cause = 'same_request_different_response'
                break
        if cause is None and len(lt) != len(rt):
            cause = 'call_count_changed'
        if cause is None and (left.get('prediction'), left.get('error', '')) != (right.get('prediction'), right.get('error', '')):
            cause = 'output_or_error_changed'
        if cause:
            changes.append(dict(user_id=key[0], sample_id=key[1], cause=cause))
    return dict(stable=not changes, samples=len(a), changes=changes)


def historical_improvement(adapter, child, parent, child_selection=None, parent_selection=None):
    """Pareto improvement: neither history partition regresses, one improves.

    Allows fitting plateaus (including perfect accuracy) to improve selection.
    Does not permit trading fitting losses for selection gains.
    """
    if child.get('errors', 0) or parent.get('errors', 0):
        return False
    fit = adapter.weighted_score(child) - adapter.weighted_score(parent)
    if child_selection is None or parent_selection is None:
        return fit > 1e-12
    if child_selection.get('errors', 0) or parent_selection.get('errors', 0):
        return False
    held = adapter.weighted_score(child_selection) - adapter.weighted_score(parent_selection)
    return fit >= -1e-12 and held >= -1e-12 and (fit > 1e-12 or held > 1e-12)
