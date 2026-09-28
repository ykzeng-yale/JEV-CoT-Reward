"""Audit-bound problem-level comparisons and honest per-policy/collection costs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analyze_screen import estimate, stratified_bootstrap_indices


def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def analyze(run,audit_path):
    audit=json.loads(audit_path.read_text())
    if audit.get('status')!='passed_guard_runtime_audit' or not audit.get('ready_for_analysis'):raise ValueError('Passed audit required')
    required={'manifest.json','outcomes.jsonl','checkpoints.jsonl','generation_events.jsonl','decisions.jsonl','skipped.jsonl'}
    if not required<=set(audit.get('input_sha256',{})):raise ValueError('Incomplete audit inventory')
    for name,h in audit['input_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((run/name).read_bytes()).hexdigest()!=h:raise ValueError('Stale audit')
    cfg=json.loads((run/'manifest.json').read_text())['config'];cps=lines(run/'checkpoints.jsonl');ids=[r['problem_id'] for r in cps]
    rows=lines(run/'outcomes.jsonl');lookup={(r['problem_id'],r['policy']):r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!={(i,p) for i in ids for p in cfg['policies']}:raise ValueError('Incomplete/duplicate grid')
    ix=stratified_bootstrap_indices([r['task']['family'] for r in cps],10000,20260930)
    events=lines(run/'generation_events.jsonl');decisions=lines(run/'decisions.jsonl');policies={};values={};censored={}
    for p in cfg['policies']:
        rr=[lookup[i,p] for i in ids];values[p]=np.array([int(r['outcome']['success']) for r in rr])
        ee=[e['generation'] for e in events if e['problem_id'] in ids and (e.get('policy')==p or e['phase']=='initial')]
        dd=[d for d in decisions if d['policy']==p]
        censored[p]=np.array([any(e['generation']['finish_reason']=='timeout' for e in events if e['problem_id']==i and (e.get('policy')==p or e['phase']=='initial')) for i in ids])
        policies[p]={'success':estimate(values[p],ix),'successes':int(values[p].sum()),'episodes':len(rr),
            'episode_generated_tokens':sum(r['episode_generated_tokens'] for r in rr),
            'processed_prompt_tokens':sum(g['prompt_tokens'] for g in ee),'service_seconds':sum(g['elapsed_seconds'] for g in ee),
            'episodes_with_timeout':int(censored[p].sum()),'timeout_calls':sum(g['finish_reason']=='timeout' for g in ee),'triggers':sum(d['trigger'] for d in dd),
            'eligible_boundaries':sum(d['boundary']>=11 and d['pool_capacity'] for d in dd)}
    contrasts={}
    for p in cfg['policies']:
        for baseline in ('continue','segmented_sham','random_branch'):
            if baseline in values and p!=baseline:
                diff=values[p]-values[baseline]
                contrasts[p+'_minus_'+baseline]={**estimate(diff,ix),'discordant_problems':int(np.count_nonzero(diff))}
                if len(ids):
                    known_p=float(np.mean(values[p]*(~censored[p])));known_b=float(np.mean(values[baseline]*(~censored[baseline])))
                    contrasts[p+'_minus_'+baseline]['timeout_completion_sensitivity']=[known_p-known_b-float(censored[baseline].mean()),known_p+float(censored[p].mean())-known_b]
    # Simultaneous bounded-difference intervals remain nondegenerate when
    # the observed paired bootstrap has zero variance. Independent problems required.
    for c in contrasts.values():
        if c['n_problems']:
            radius=math.sqrt(2*math.log(2*len(contrasts)/.05)/c['n_problems'])
            c['simultaneous_hoeffding_95']=[max(-1.,c['mean']-radius),min(1.,c['mean']+radius)]
    return {'status':'audited_development_only','scheduled_problems':cfg['problems'],'eligible_problems':len(ids),
        'skipped_problems':len(lines(run/'skipped.jsonl')),'policies':policies,'paired_contrasts':contrasts,
        'collection_generated_tokens':sum(e['generation']['generated_tokens'] for e in events),
        'collection_service_seconds':sum(e['generation']['elapsed_seconds'] for e in events),
        'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'analysis_dependency_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('analyze_guard_qualification.py','analyze_screen.py')},
        'limitations':['Problem-stratified descriptive bootstrap; small samples and zero discordance cannot establish equivalence.',
            'Hoeffding intervals assume independent original problems and frozen policies; simultaneous over reported contrasts, conservative and conditional on the task-generation/enrollment design, not an OOD guarantee.',
            'Per-policy costs include the shared initial prefix once; collection costs count each actual call once.',
            'GPU allocation time and infrastructure preflight are separate from episode service time.',
            'Opportunity-rate matching is not exact intervention-count, latency or compute matching.',
            'Timeout-completion sensitivity is a finite-sample worst-case envelope, not a confidence interval; it assumes non-timeout outcomes would stay unchanged under the hypothetical completion regime.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    a.output.write_text(json.dumps(analyze(a.run,a.audit),indent=2)+'\n')
