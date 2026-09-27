"""Gold-free Jev candidate choice; private responses and explicit acquisition cost."""
from .branch_selectors import local_prompt
from .jev import RESERVE_USD


def request(task_prompt, retained_text, views):
    # Identical observable text and selection instruction as the local selector.
    return local_prompt(task_prompt, retained_text, views), {
        'candidate': {'type': 'choice',
            'instructions': 'Select the candidate index using the task and selection instructions in state.',
            'criteria': {str(i): f'Candidate next segment with index {i} is the best choice.' for i in range(len(views))}}}


def select(client, task_prompt, retained_text, views):
    import time
    state, questions = request(task_prompt, retained_text, views)
    start = time.monotonic()
    try:
        result = client.evaluate(state, questions)
        answer = result['response']['answers']['candidate']
        return {'choice': int(answer['choice']), 'error': None, 'request': {'state': state, 'questions': questions},
                'result': result, 'accounted_usd': result['input_cost_usd'],
                'input_tokens': result['response']['usage']['input_tokens'],
                'acquisition_seconds': time.monotonic()-start}
    except Exception as exc:
        # No retry or response/error body. Failure remains a policy outcome.
        return {'choice': None, 'error': type(exc).__name__, 'request': {'state': state, 'questions': questions},
                'accounted_usd': RESERVE_USD, 'input_tokens': None,
                'acquisition_seconds': time.monotonic()-start,
                'cost_note': 'Conservative reservation; authoritative cumulative charge is in the central ledger.'}
