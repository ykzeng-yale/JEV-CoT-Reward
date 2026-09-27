"""Independent checks of prospective assignment and per-policy cost accounting.

This layer does not certify token replay or model predictions; the complete
run audit must additionally check those against frozen source/model artifacts.
"""
import hashlib
import json
import math
from collections import Counter

POLICIES = {'always_continue', 'cheap_tfidf', 'cheap_tfidf_plus_jev', 'cheap_tfidf_plus_local'}
ACTIONS = {'continue','repair','branch'}


def assignment_audit(schedule, checkpoints, decision_bytes, outcomes, events):
    ids = [x['task']['id'] for x in schedule]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate schedule problem')
    decisions, hashes = {}, {}
    for line in decision_bytes.splitlines(keepends=True):
        record = json.loads(line)
        pid = record['problem_id']
        if pid in decisions:
            raise ValueError('Duplicate decision')
        decisions[pid] = record
        hashes[pid] = hashlib.sha256(line).hexdigest()
    cps = {x['problem_id']:x for x in checkpoints}
    if len(cps) != len(checkpoints) or set(cps) != set(ids) or set(decisions) != set(ids):
        raise ValueError('Incomplete enrolled checkpoint or decision inventory')
    schedule_by_id = {x['task']['id']:x for x in schedule}
    seen, policy_count = set(), Counter()
    for pid, decision in decisions.items():
        if set(decision['decisions']) != POLICIES or set(decision['acquisition_costs']) != POLICIES:
            raise ValueError('Incomplete policy assignment')
        if decision['decisions']['always_continue'] != 'continue' or not set(decision['decisions'].values()) <= ACTIONS:
            raise ValueError('Invalid assigned action')
        if decision['seed'] != schedule_by_id[pid]['continuation_seed']:
            raise ValueError('Decision seed mismatch')
        if decision['checkpoint_sha256'] != cps[pid]['sha256']:
            raise ValueError('Decision checkpoint mismatch')
        if not math.isfinite(decision['recorded_unix']):
            raise ValueError('Invalid decision timestamp')
    for event in events:
        if event.get('phase') == 'continuation':
            pid = event['problem_id']
            if pid not in decisions or event['started_unix'] < decisions[pid]['recorded_unix']:
                raise ValueError('Continuation started before policy decisions')
            if event['action'] not in decisions[pid]['decisions'].values():
                raise ValueError('Unselected action generated')
    for row in outcomes:
        pid, action = row['problem_id'], row['action']
        if pid not in decisions or (pid,action) in seen:
            raise ValueError('Unknown or duplicate action outcome')
        seen.add((pid,action))
        d, initial = decisions[pid], cps[pid]['initial']
        expected = {p for p,a in d['decisions'].items() if a==action}
        if not expected or set(row['policies']) != expected or len(row['policies']) != len(expected):
            raise ValueError('Shared outcome policy mapping mismatch')
        if row['decision_record_sha256'] != hashes[pid] or row['seed'] != d['seed']:
            raise ValueError('Outcome decision hash or seed mismatch')
        if row['checkpoint_sha256'] != d['checkpoint_sha256']:
            raise ValueError('Outcome checkpoint mismatch')
        if set(row['hypothetical_deployment_costs']) != expected:
            raise ValueError('Missing per-policy costs')
        for policy in expected:
            policy_count[policy] += 1
            cost = row['hypothetical_deployment_costs'][policy]
            for key,a,b in [('generated_tokens','generated_tokens','generated_tokens'),
                            ('prompt_tokens_processed','prompt_tokens','prompt_tokens_processed'),
                            ('model_service_seconds','elapsed_seconds','elapsed_seconds')]:
                value = initial[a] + row[b]
                if not math.isfinite(value) or value<0 or not math.isclose(cost[key],value,rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('Full hypothetical deployment cost mismatch')
            if cost['acquisition'] != d['acquisition_costs'][policy]:
                raise ValueError('Acquisition cost changed after decision')
    expected_pairs = {(pid,a) for pid,d in decisions.items() for a in d['decisions'].values()}
    if seen != expected_pairs or any(policy_count[p] != len(ids) for p in POLICIES):
        raise ValueError('Missing selected outcomes or enrolled policy episodes')
    return {'problems':len(ids),'unique_action_outcomes':len(outcomes),
            'policy_episodes':sum(policy_count.values()),'assignment_and_cost_checks':'passed',
            'scope':'Does not certify token replay, feature provenance, prediction correctness or terminal verification'}
