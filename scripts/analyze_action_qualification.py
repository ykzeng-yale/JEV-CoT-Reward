#!/usr/bin/env python3
"""Descriptive development action comparisons after a matching passed audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_screen import stratified_bootstrap_indices, estimate


def split_repeat_diagnostic(lookup, ids, actions, repeats):
    """Outcome-informed diagnostic only; disjoint labels select/evaluate."""
    if repeats != 4:return []
    reports=[]
    for train,test in [([0,1],[2,3]),([2,3],[0,1])]:
        selected=[];baseline=[];chosen=[]
        for pid in ids:
            best=max(actions,key=lambda a:sum(lookup[pid,a,r]['outcome']['success'] for r in train))
            chosen.append(best)
            selected.append(float(np.mean([lookup[pid,best,r]['outcome']['success'] for r in test])))
            baseline.append(float(np.mean([lookup[pid,'continue',r]['outcome']['success'] for r in test])))
        reports.append({'selection_repeats':train,'evaluation_repeats':test,
            'selected_success':float(np.mean(selected)),'continue_success':float(np.mean(baseline)),
            'paired_difference':float(np.mean(np.asarray(selected)-baseline)),
            'selected_action_counts':dict(Counter(chosen)),
            'scope':'Outcome-informed diagnostic, not deployable policy or oracle bound; reversed splits overlap and are not independent.'})
    return reports


def analyze(run,audit_path):
    audit=json.loads(audit_path.read_text())
    if audit.get('status')!='passed_action_qualification_audit' or audit.get('ready_for_analysis') is not True:
        raise ValueError('Passed audit required')
    hashes=audit.get('input_sha256',{})
    if not {'outcomes.jsonl','checkpoints.jsonl','schedule.json','manifest.json','generation_events.jsonl'}<=set(hashes):
        raise ValueError('Incomplete audit inventory')
    for name,digest in hashes.items():
        if Path(name).name!=name or hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Inputs changed after audit')
    config=json.loads((run/'manifest.json').read_text())['config']
    rows=[json.loads(s) for s in (run/'outcomes.jsonl').read_text().splitlines()]
    cps=[json.loads(s) for s in (run/'checkpoints.jsonl').read_text().splitlines()]
    ids=[cp['problem_id'] for cp in cps];families=[cp['task']['family'] for cp in cps]
    if not ids:raise ValueError('No eligible checkpoints; report enrollment only')
    lookup={(r['problem_id'],r['action'],r['repeat']):r for r in rows}
    expected={(pid,a,r) for pid in ids for a in config['actions'] for r in range(config['repeats'])}
    if len(lookup)!=len(rows) or set(lookup)!=expected:raise ValueError('Incomplete/duplicate outcome grid')
    indices=stratified_bootstrap_indices(families,10000,20260928)
    values={a:np.asarray([np.mean([lookup[pid,a,r]['outcome']['success'] for r in range(config['repeats'])]) for pid in ids]) for a in config['actions']}
    actions={}
    initial={cp['problem_id']:cp['initial'] for cp in cps}
    for action in config['actions']:
        selected=[r for r in rows if r['action']==action]
        actions[action]={'successes':sum(r['outcome']['success'] for r in selected),'episodes':len(selected),
            'problem_weighted_success':estimate(values[action],indices),
            'family_success':{f:float(values[action][np.asarray(families)==f].mean()) for f in set(families)},
            'mean_episode_generated_tokens':float(np.mean([r['generated_tokens']+initial[r['problem_id']]['generated_tokens'] for r in selected])),
            'mean_episode_processed_prompt_tokens':float(np.mean([r['prompt_tokens_processed']+initial[r['problem_id']]['prompt_tokens'] for r in selected])),
            'mean_removed_tokens':float(np.mean([r['preparation']['removed_tokens'] for r in selected])),
            'mean_inserted_instruction_tokens':float(np.mean([r['preparation']['inserted_tokens'] for r in selected])),
            'last_finish_reason_counts':dict(Counter(r['calls'][-1]['finish_reason'] if r['calls'] else 'no_call' for r in selected)),
            'outcome_reason_counts':dict(Counter(r['outcome']['reason'] for r in selected)),
            'mean_episode_model_service_seconds':float(np.mean([r['elapsed_seconds']+initial[r['problem_id']]['elapsed_seconds'] for r in selected]))}
    contrasts={}
    for left,right in [(a,'continue') for a in config['actions'] if a!='continue']+[('suffix_repair','recheck'),('segment_repair','suffix_repair'),('recheck','sham'),('segment_repair','sham')]:
        if left not in values or right not in values:continue
        d=values[left]-values[right]
        contrasts[left+'_minus_'+right]={**estimate(d,indices),
            'warning':'Descriptive problem-clustered comparison; exploratory multiplicity, no automatic winner selection.'}
    return {'status':'audited_development_action_qualification','enrolled_problems':config['problems'],
        'eligible_problems':len(ids),'episodes':len(rows),'actions':actions,'paired_contrasts':contrasts,
        'split_repeat_diagnostic':split_repeat_diagnostic(lookup,ids,config['actions'],config['repeats']),
        'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Conditional on online checkpoint eligibility; no full-policy deployment claim.',
            'Few repeats per action give noisy state-specific effects; selection/evaluation splitting does not remove all uncertainty.',
            'Generator-token allowances matched; prefill, actual length and runtime differ.',
            'No Jev acquisition, fitted controller, novel task-family transfer or scientific discovery tested.',
            'Historical v1 is not a randomized budget comparator; do not pool.',
            'A degenerate empirical bootstrap does not establish zero uncertainty.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    a.output.write_text(json.dumps(analyze(a.run,a.audit),indent=2)+'\n')
