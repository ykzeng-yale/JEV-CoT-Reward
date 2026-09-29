"""Audit natural labels, Jev request hashes, measured costs and sample inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.jev import MODEL,PRICE_PER_MILLION,canonical
from build_jev_natural_arithmetic import build


def audit(config,run):
    m=json.loads(config.read_text());status=json.loads((run/'status.json').read_text())
    if m!=build():raise ValueError('Frozen natural selection changed')
    if hashlib.sha256(config.read_bytes()).hexdigest()!=status['manifest_sha256']:raise ValueError('Manifest changed')
    rows=[json.loads(x) for x in (run/'measurements.jsonl').read_text().splitlines()]
    if status['completed_requests']!=len(rows) or status['status'] not in ('complete','failed'):raise ValueError('Bad run status')
    if status['status']=='complete' and len(rows)!=48:raise ValueError('Incomplete nominal run')
    if status['status']=='failed' and not any(g['status']=='unresolved' for g in status['ledger_after']['groups']):raise ValueError('Failure not reserved')
    for i,row in enumerate(rows):
        if row['index']!=i or not 0<=row['noul']<=1:raise ValueError('Order/score mismatch')
        item=m['rows'][i];a,b,c=item['claim']
        if item['label']!=(a+b==c):raise ValueError('Natural label mismatch')
        payload={'model':MODEL,'state':item['state'],'questions':{'equality_true':m['question']}}
        if row['request_sha256']!=hashlib.sha256(canonical(payload).encode()).hexdigest():raise ValueError('Request hash mismatch')
        if abs(row['input_cost_usd']-row['input_tokens']*PRICE_PER_MILLION/1e6)>1e-12:raise ValueError('Cost mismatch')
    y=np.array([m['rows'][i]['label'] for i in range(len(rows))]);p=np.array([row['noul'] for row in rows]);has_both=len(set(y))==2
    return {'status':'audited_natural_addition_sensor','run_status':status['status'],'planned':48,'observed':len(rows),
        'labels_observed':{'true':int(y.sum()),'false':int(len(y)-y.sum())},
        'labels_missing':{'true':sum(x['label'] for x in m['rows'][len(rows):]),'false':sum(not x['label'] for x in m['rows'][len(rows):])},
        'auroc':float(roc_auc_score(y,p)) if has_both else None,
        'sensitivity_at_0_5':float(np.mean(p[y]>=.5)) if y.any() else None,
        'specificity_at_0_5':float(np.mean(p[~y]<.5)) if (~y).any() else None,
        'mean_true':float(np.mean(p[y])) if y.any() else None,'mean_false':float(np.mean(p[~y])) if (~y).any() else None,
        'cached_requests':sum(row['cache_hit'] for row in rows),
        'measured_request_cost_usd':sum(row['input_cost_usd'] for row in rows),
        'ledger_accounted_usd':status['ledger_after']['accounted_usd'],
        'config_sha256':status['manifest_sha256'],'measurements_sha256':hashlib.sha256((run/'measurements.jsonl').read_bytes()).hexdigest(),
        'scope':'Natural explicit additions in public development prefixes; no downstream intervention outcome or controller value identified.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    a.output.write_text(json.dumps(audit(a.config,a.run),indent=2)+'\n')
