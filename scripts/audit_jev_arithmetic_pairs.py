"""Independent paired-label, request-identity and ledger audit; report summary only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.jev import canonical,MODEL,PRICE_PER_MILLION
from build_jev_arithmetic_pairs import build


def audit(config,run):
    m=json.loads(config.read_text());s=json.loads((run/'status.json').read_text())
    if s['status'] not in ('complete','failed') or s['completed_requests'] not in (29,32):raise ValueError('Unexpected API study state')
    if s['status']=='failed' and (s['completed_requests']!=29 or s.get('error_type')!='RuntimeError' or not any(g['status']=='unresolved' and g['attempts']>=1 for g in s['ledger_after']['groups'])):raise ValueError('Unaccounted failure')
    if hashlib.sha256(config.read_bytes()).hexdigest()!=s['manifest_sha256']:raise ValueError('Manifest mismatch')
    if m!=build():raise ValueError('Frozen source derivation mismatch')
    rows=[json.loads(x) for x in (run/'measurements.jsonl').read_text().splitlines()]
    if len(rows)!=s['completed_requests']:raise ValueError('Request count mismatch')
    seen=set();pairs={}
    for row in rows:
        i=row['pair'];kind=row['kind']
        if not 0<=i<16 or kind not in ('correct','corrupted') or (i,kind) in seen:raise ValueError('Duplicate/invalid measurement')
        seen.add((i,kind))
        p=m['pairs'][i];state=dict(p['state'])
        if kind=='corrupted':state['latest_segment']=p['altered_segment']
        payload={'model':MODEL,'state':state,'questions':{'equality_true':m['question']}}
        request=hashlib.sha256(canonical(payload).encode()).hexdigest()
        if row['request_sha256']!=request:raise ValueError('Request identity mismatch')
        if not 0<=row['noul']<=1 or row['input_tokens']<0:raise ValueError('Invalid response')
        if abs(row['input_cost_usd']-row['input_tokens']*PRICE_PER_MILLION/1e6)>1e-12:raise ValueError('Cost mismatch')
        a,b,c=p['claim'];assert a+b==c
        if p['altered_segment']==p['state']['latest_segment']:raise ValueError('No actual corruption')
        pairs.setdefault(i,{})[kind]=row['noul']
    if s['status']=='complete' and len(seen)!=32:raise ValueError('Missing arm')
    if s['status']=='failed' and (set(seen)!={(i,k) for i in range(14) for k in ('correct','corrupted')}|{(14,'correct' if m['pairs'][14]['order'][0]==0 else 'corrupted')}):raise ValueError('Unexpected partial selection')
    complete=16 if s['status']=='complete' else 14
    diff=np.array([pairs[i]['correct']-pairs[i]['corrupted'] for i in range(complete)])
    rng=np.random.default_rng(20260928);boot=diff[rng.integers(complete,size=(10000,complete))].mean(axis=1)
    return {'status':'audited_jev_arithmetic_sensor_diagnostic','planned_pairs':16,'complete_pairs':complete,'successful_requests':len(rows),'run_status':s['status'],
            'mean_correct_probability':float(np.mean([pairs[i]['correct'] for i in range(complete)])),
            'mean_corrupted_probability':float(np.mean([pairs[i]['corrupted'] for i in range(complete)])),
            'mean_paired_difference':float(diff.mean()),'positive_pairs':int(np.sum(diff>0)),
            'ties':int(np.sum(diff==0)),'descriptive_pair_bootstrap_95':[float(x) for x in np.quantile(boot,[.025,.975])],
            'paired_values':[[float(pairs[i]['correct']),float(pairs[i]['corrupted'])] for i in range(complete)],
            'measured_input_cost_usd':float(sum(r['input_cost_usd'] for r in rows)),
            'ledger_after_accounted_usd':s['ledger_after']['accounted_usd'],
            'manifest_sha256':s['manifest_sha256'],
            'measurements_sha256':hashlib.sha256((run/'measurements.jsonl').read_bytes()).hexdigest(),
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Synthetic matched arithmetic defect in public natural prefixes; no action-value, adaptive-policy or OOD efficacy claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    a.output.write_text(json.dumps(audit(a.config,a.run),indent=2)+'\n')
