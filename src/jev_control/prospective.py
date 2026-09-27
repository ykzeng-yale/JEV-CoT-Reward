"""Prospective ordering and shared-action accounting, independent of backend."""
import hashlib
import json
import os
from pathlib import Path
import time

POLICIES = ('always_continue', 'cheap_tfidf', 'cheap_tfidf_plus_jev', 'cheap_tfidf_plus_local')
ACTIONS = ('continue', 'repair', 'branch')


def durable_record(path, record):
    encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode()
    with Path(path).open('ab') as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def execute_selected(problem_id, decisions, seed, initial, decision_path, outcome_path,
                     run_action, *, checkpoint_sha256, acquisition_costs):
    """Persist every decision before the first action; run each chosen action once.

    run_action receives an action and the frozen common seed. It must enforce
    the remaining generator envelope and return measured continuation costs.
    Each policy receives its full hypothetical cost despite shared collection.
    """
    if set(decisions) != set(POLICIES) or any(a not in ACTIONS for a in decisions.values()):
        raise ValueError('Complete valid policy decisions required')
    if decisions['always_continue'] != 'continue':
        raise ValueError('Continue comparator must continue')
    if set(acquisition_costs) != set(POLICIES):
        raise ValueError('Explicit acquisition cost for each policy required')
    existing = Path(decision_path)
    if existing.exists():
        for line in existing.read_text().splitlines():
            if json.loads(line)['problem_id'] == problem_id:
                raise ValueError('Problem decisions already recorded; refusing replay')
    record = {'problem_id': problem_id, 'checkpoint_sha256': checkpoint_sha256,
              'decisions': dict(decisions), 'seed': seed, 'recorded_unix': time.time(),
              'acquisition_costs': acquisition_costs}
    decision_hash = durable_record(decision_path, record)
    outputs = []
    for action in ACTIONS:
        policies = [p for p in POLICIES if decisions[p] == action]
        if not policies:
            continue
        row = run_action(action, seed)
        if row['action'] != action or row['seed'] != seed:
            raise ValueError('Backend returned an unselected action or seed')
        measurements = {}
        for policy in policies:
            measurements[policy] = {
                'generated_tokens': initial['generated_tokens'] + row['generated_tokens'],
                'prompt_tokens_processed': initial['prompt_tokens'] + row['prompt_tokens_processed'],
                'model_service_seconds': initial['elapsed_seconds'] + row['elapsed_seconds'],
                'acquisition': acquisition_costs[policy],
            }
        output = {**row, 'problem_id': problem_id, 'checkpoint_sha256': checkpoint_sha256,
                  'decision_record_sha256': decision_hash, 'policies': policies,
                  'hypothetical_deployment_costs': measurements,
                  'collection_sharing': 'One actual continuation per selected action; full cost per policy'}
        durable_record(outcome_path, output)
        outputs.append(output)
    return outputs
