"""Independent qualification of pinned GUARD decision semantics, not a runner.

Reference: ZHUWEI-hub/GUARD at 8d2bc7af, eval/math_eval_guard.py.
No benchmark result or complete reproduction is implied.
"""
import math


def entropy_trigger(boundary_history, remaining_tokens, quantile=.90):
    """History includes current boundary's predictive entropy; strict comparison."""
    if not 0 <= quantile < 1:
        raise ValueError('Quantile must be in [0,1)')
    if any(not math.isfinite(x) or x < 0 for x in boundary_history):
        raise ValueError('Finite nonnegative entropy required')
    if len(boundary_history) < 5 or remaining_tokens <= 200:
        return False
    threshold=sorted(boundary_history)[int(len(boundary_history)*quantile)]
    return boundary_history[-1] > threshold


def branch_score(onset_entropy, candidate_entropies, completion_marker=False):
    if not candidate_entropies:
        return -math.inf
    if not math.isfinite(onset_entropy) or any(not math.isfinite(x) or x < 0 for x in candidate_entropies):
        raise ValueError('Invalid entropy')
    return onset_entropy-sum(candidate_entropies)/len(candidate_entropies)+(10. if completion_marker else 0.)


# Temperatures are absolute, despite the upstream variable's "adjust" name.
BRANCH_TEMPERATURES=(0.,.6,1.5)
