"""Study-stratified, problem-level descriptive synthesis of two audited static studies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def records(run,audit):
    proof=json.loads(audit.read_text())
    if proof.get('status')!='passed_action_qualification_audit' or not proof.get('ready_for_analysis'):
        raise ValueError('Passed complete audit required')
    for name,digest in proof['input_sha256'].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:raise ValueError('Stale audit')
    config=json.loads((run/'manifest.json').read_text())['config']
    rows=[json.loads(x) for x in (run/'outcomes.jsonl').read_text().splitlines()]
    ids=[r['problem_id'] for r in map(json.loads,(run/'checkpoints.jsonl').read_text().splitlines())]
    if len(ids)!=config['problems'] or len(set(ids))!=len(ids):raise ValueError('Enrollment mismatch')
    out=[]
    for pid in ids:
        z={}
        for arm in ('continue','sham'):
            rr=[r for r in rows if r['problem_id']==pid and r['action']==arm]
            if len(rr)!=config['repeats'] or len({r['repeat'] for r in rr})!=len(rr):raise ValueError('Incomplete arm grid')
            z[arm]=float(np.mean([r['outcome']['success'] for r in rr]))
        out.append(z)
    return out


def synth(studies):
    blocks=[]
    for name,run,audit in studies:
        rr=records(run,audit)
        d=np.array([r['sham']-r['continue'] for r in rr]);blocks.append((name,d,rr))
    rng=np.random.default_rng(20260930);draws=20000
    # Equal weight for each independently enrolled original problem. Resample within studies.
    sampled=np.zeros(draws)
    for _,d,_ in blocks:sampled+=d[rng.integers(len(d),size=(draws,len(d)))].sum(axis=1)
    n=sum(len(d) for _,d,_ in blocks);sampled/=n
    return {'status':'descriptive_audit_bound_two_study_synthesis','n_problems':n,
        'paired_mean':float(sum(d.sum() for _,d,_ in blocks)/n),
        'stratified_problem_bootstrap_95':[float(x) for x in np.quantile(sampled,[.025,.975])],
        'studies':[{'name':name,'n_problems':len(d),'paired_mean':float(d.mean()),'discordant_problems':int(np.count_nonzero(d)),
            'continue_mean':float(np.mean([r['continue'] for r in rr])),'sham_mean':float(np.mean([r['sham'] for r in rr]))} for name,d,rr in blocks],
        'audit_sha256':{name:hashlib.sha256(audit.read_bytes()).hexdigest() for name,_,audit in studies},
        'limitations':['Exploratory development synthesis, not a new randomized prospective study or preregistered pooled test.',
            '48 and 24 original problems have unequal repeats; each problem receives equal weight.',
            'Problem bootstrap assumes study samples represent target problems; no arbitrary family/OOD guarantee.',
            'Semantic judge and adaptive controller absent. A small static effect need not justify further compute.']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    root=Path('runs');result=Path('results')
    studies=[('repair_v3',root/'action-replication-v3-20260928',result/'action_replication_v3_audit.json'),
        ('sham_v4',root/'sham-replication-v4-20260928',result/'sham_replication_v4_audit.json')]
    a.output.write_text(json.dumps(synth(studies),indent=2)+'\n')
