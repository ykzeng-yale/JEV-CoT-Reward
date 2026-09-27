#!/usr/bin/env python3
"""Problem-paired analysis of a complete, independently audited prospective run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.prospective import POLICIES
from analyze_screen import stratified_bootstrap_indices, estimate


def read_lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def analyze(run, integrity, draws=10000):
    audit=json.loads(integrity.read_text())
    if audit.get('status')!='passed_prospective_audit' or audit.get('ready_for_analysis') is not True:
        raise ValueError('Passed prospective audit required')
    required={'manifest.json','summary.json','schedule.json','checkpoints.jsonl','decisions.jsonl',
              'outcomes.jsonl','predictions.jsonl','local_judge.jsonl','generation_started.jsonl','generation_events.jsonl','rubric.json'}
    if not required<=set(audit.get('input_sha256',{})):
        raise ValueError('Audit input inventory incomplete')
    for name,digest in audit['input_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Inputs changed after audit')
    schedule=json.loads((run/'schedule.json').read_text())
    rows=read_lines(run/'outcomes.jsonl')
    ids=[s['task']['id'] for s in schedule];families=[s['task']['family'] for s in schedule]
    lookup={}
    for row in rows:
        for policy in row['policies']:
            key=(row['problem_id'],policy)
            if key in lookup:raise ValueError('Duplicate policy episode')
            lookup[key]=row
    if set(lookup)!={(p,policy) for p in ids for policy in POLICIES}:
        raise ValueError('Incomplete prospective episode grid')
    indices=stratified_bootstrap_indices(families,draws,20260927)
    values={policy:np.asarray([int(lookup[p,policy]['outcome']['success']) for p in ids]) for policy in POLICIES}
    policies={}
    for policy in POLICIES:
        selected=[lookup[p,policy] for p in ids]
        costs=[r['hypothetical_deployment_costs'][policy] for r in selected]
        acquisition=[c['acquisition'] for c in costs]
        policies[policy]={'successes':int(values[policy].sum()),'success':estimate(values[policy],indices),
            'action_counts':dict(Counter(r['action'] for r in selected)),
            'mean_generator_tokens':float(np.mean([c['generated_tokens'] for c in costs])),
            'mean_generator_prompt_tokens':float(np.mean([c['prompt_tokens_processed'] for c in costs])),
            'mean_generator_service_seconds':float(np.mean([c['model_service_seconds'] for c in costs])),
            'mean_acquisition_usd':float(np.mean([c['usd'] for c in acquisition])) if all(c['usd'] is not None for c in acquisition) else None,
            'mean_acquisition_service_seconds':float(np.mean([c['service_seconds'] for c in acquisition])) if all(c['service_seconds'] is not None for c in acquisition) else None,
            'mean_acquisition_generated_tokens':float(np.mean([c['generated_tokens'] for c in acquisition]))}
    contrasts={}
    for left,right in [('cheap_tfidf_plus_jev','cheap_tfidf'),('cheap_tfidf_plus_local','cheap_tfidf'),
                       ('cheap_tfidf_plus_jev','cheap_tfidf_plus_local'),*[(p,'always_continue') for p in POLICIES if p!='always_continue']]:
        diff=values[left]-values[right]
        contrasts[left+'_minus_'+right]={**estimate(diff,indices),'discordant_problems':int((diff!=0).sum())}
    family_results={f:{p:float(v[np.asarray(families)==f].mean()) for p,v in values.items()} for f in sorted(set(families))}
    events=read_lines(run/'generation_events.jsonl')
    return {'status':'completed_small_prospective_diagnostic','independent_problems':len(ids),
            'policy_episodes':len(lookup),'actual_shared_action_continuations':len(rows),
            'policies':policies,'paired_contrasts':contrasts,'family_success':family_results,
            'collection_generated_tokens':sum(e['generation']['generated_tokens'] for e in events),
            'collection_model_service_seconds':sum(e['generation']['elapsed_seconds'] for e in events),
            'audit_sha256':hashlib.sha256(integrity.read_bytes()).hexdigest(),
            'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limitations':['24 problems and one trajectory each cannot establish equivalence or small benefits.',
                          'Policies were frozen before these tasks; this is one-checkpoint control, not sequential adaptation.',
                          'Identical actions share continuation outcomes; inference and resampling are paired by original problem.',
                          'Bootstrap ranges are descriptive; report all contrasts, without post-test policy selection.',
                          'Generator tokens are matched; acquisition and prefill costs are additional.',
                          'Summed model service is not end-to-end latency or a matched wall-clock deployment test.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--integrity-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    report=analyze(a.run,a.integrity_report)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'problems':report['independent_problems']}))
