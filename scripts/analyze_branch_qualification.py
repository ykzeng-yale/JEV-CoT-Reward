#!/usr/bin/env python3
"""Problem-clustered common-pool diagnosis after a matching independent audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_screen import estimate,stratified_bootstrap_indices


def read_lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def analyze(run,audit_path):
    audit=json.loads(audit_path.read_text())
    if audit.get('status')!='passed_branch_qualification_audit' or audit.get('ready_for_analysis') is not True:
        raise ValueError('Passed audit required')
    required={'outcomes.jsonl','schedule.json','manifest.json','decisions.jsonl','candidates.jsonl','generation_events.jsonl','checkpoints.jsonl'}
    if not required<=set(audit.get('input_sha256',{})):raise ValueError('Incomplete audit inventory')
    for name,digest in audit['input_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:raise ValueError('Stale audit')
    config=json.loads((run/'manifest.json').read_text())['config']
    cps=read_lines(run/'checkpoints.jsonl');ids=[r['problem_id'] for r in cps];families=[r['task']['family'] for r in cps]
    if not ids:raise ValueError('No eligible checkpoints; report enrollment only')
    rows=read_lines(run/'outcomes.jsonl');lookup={}
    policies=['continue',*config['selectors']]
    for row in rows:
        for policy in row['selectors']:
            key=(row['problem_id'],row['repeat'],policy)
            if key in lookup:raise ValueError('Duplicate policy episode')
            lookup[key]=row
    expected={(pid,r,p) for pid in ids for r in range(config['repeats']) for p in policies}
    if set(lookup)!=expected:raise ValueError('Incomplete policy episode grid')
    indices=stratified_bootstrap_indices(families,10000,20260929)
    values={p:np.asarray([np.mean([lookup[pid,r,p]['outcome']['success'] for r in range(config['repeats'])]) for pid in ids]) for p in policies}
    results={}
    for policy in policies:
        selected=[lookup[pid,r,policy] for pid in ids for r in range(config['repeats'])]
        acquire=policy=='local_semantic'
        results[policy]={'successes':sum(r['outcome']['success'] for r in selected),'episodes':len(selected),
            'problem_weighted_success':estimate(values[policy],indices),
            'family_success':{f:float(values[policy][np.asarray(families)==f].mean()) for f in sorted(set(families))},
            'mean_episode_generator_tokens':float(np.mean([r['episode_generated_tokens'] for r in selected])),
            'mean_selector_generated_tokens':float(np.mean([r['selector_generated_tokens'] if acquire else 0 for r in selected])),
            'mean_total_processed_prompt_tokens':float(np.mean([r['episode_prompt_tokens']+(r['selector_prompt_tokens'] if acquire else 0) for r in selected])),
            'mean_total_model_service_seconds':float(np.mean([r['episode_service_seconds']+(r['selector_service_seconds'] if acquire else 0) for r in selected]))}
    if config.get('jev'):
        for policy in policies:
            selected=[lookup[pid,r,policy] for pid in ids for r in range(config['repeats'])]
            hosted=policy=='jev_semantic'
            results[policy]['mean_jev_accounted_usd']=float(np.mean([r.get('jev_accounted_usd',0) if hosted else 0 for r in selected]))
            results[policy]['mean_jev_acquisition_seconds']=float(np.mean([r.get('jev_acquisition_seconds',0) if hosted else 0 for r in selected]))
            results[policy]['mean_jev_input_tokens']=float(np.mean([r.get('jev_input_tokens') or 0 if hosted else 0 for r in selected]))
    contrasts={}
    pairs=[(p,'continue') for p in policies[1:]]+[(p,'uniform') for p in policies[2:]]
    if config.get('jev'):pairs += [('jev_semantic','likelihood'),('jev_semantic','local_semantic')]
    for left,right in pairs:
        contrasts[left+'_minus_'+right]={**estimate(values[left]-values[right],indices),
            'warning':'Exploratory development comparison; no nominal coverage or post-selection efficacy claim.'}
    decisions=read_lines(run/'decisions.jsonl');candidates=read_lines(run/'candidates.jsonl')
    agreement={p:sum(d['choices'][p]==d['choices']['uniform'] for d in decisions) for p in config['selectors']}
    split=[]
    by_candidate={(r['problem_id'],r['repeat'],r['candidate']):r for r in rows if r['candidate'] is not None}
    # This is explicitly an outcome-informed development diagnostic, not a policy.
    half=config['repeats']//2
    splits=[(list(range(half)),list(range(half,config['repeats']))),
            (list(range(half,config['repeats'])),list(range(half)))]
    for selection,evaluation in splits:
        chosen=[];uniform=[]
        for pid in ids:
            winner=max(range(config['candidate_count']),key=lambda j:sum(int(by_candidate[pid,r,j]['outcome']['success']) for r in selection))
            chosen.append(np.mean([by_candidate[pid,r,winner]['outcome']['success'] for r in evaluation]))
            uniform.append(np.mean([lookup[pid,r,'uniform']['outcome']['success'] for r in evaluation]))
        split.append({'selection_repeats':selection,'evaluation_repeats':evaluation,
            'selected_candidate_success':float(np.mean(chosen)),'uniform_success':float(np.mean(uniform)),
            'interpretation':'Noisy outcome-informed candidate selection; overlapping splits, not deployable and not an oracle bound.'})
    events=read_lines(run/'generation_events.jsonl')
    return {'status':'audited_development_branch_qualification','enrolled_problems':config['problems'],
        'eligible_problems':len(ids),'unique_outcomes':len(rows),'hypothetical_policy_episodes':len(lookup),
        'policies':results,'paired_contrasts':contrasts,'agreement_with_uniform_by_problem':agreement,
        'jev_failure_count':sum('jev_semantic' in d['failures'] for d in decisions),
        'local_failure_count':sum('local_semantic' in d['failures'] for d in decisions),
        'candidate_unique_text_counts':{pid:len({c['generation']['text'] for c in candidates if c['problem_id']==pid}) for pid in ids},
        'split_repeat_diagnostic':split,
        'collection_generated_tokens':sum(e['generation']['generated_tokens'] for e in events if e['status']=='complete'),
        'collection_service_seconds':sum(e['generation']['elapsed_seconds'] for e in events if e['status']=='complete'),
        'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Shared candidate pool and continuations couple policies; independent unit is original problem.',
            'Generator caps do not match total compute; local judge acquisition is separately charged.',
            'Shadow continuations for unselected candidates are collection cost, not free deployed evidence.',
            'The complete GUARD trigger, prompts and scheduling are not reproduced.',
            'Direct selector development study; no confirmatory Jev efficacy, transfer, sequential control or training tested.',
            'A degenerate empirical bootstrap does not establish equivalence.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    a.output.write_text(json.dumps(analyze(a.run,a.audit),indent=2)+'\n')
