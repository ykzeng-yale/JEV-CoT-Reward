"""Descriptive stratified bootstrap for a completed audited Jev sensor study."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def compute(manifest,measurements,audit):
    a=json.loads(audit.read_text())
    if a['status']!='audited_natural_addition_sensor' or a['run_status']!='complete' or a['observed']!=48:raise ValueError('Complete audit required')
    if hashlib.sha256(measurements.read_bytes()).hexdigest()!=a['measurements_sha256']:raise ValueError('Stale audit')
    m=json.loads(manifest.read_text());rows=[json.loads(x) for x in measurements.read_text().splitlines()]
    y=np.array([bool(m['rows'][i]['label']) for i in range(48)]);p=np.array([r['noul'] for r in rows]);rng=np.random.default_rng(20260930)
    positive=np.flatnonzero(y);negative=np.flatnonzero(~y);draws=10000
    auc=[];specificity=[]
    for _ in range(draws):
        ix=np.concatenate([rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)])
        auc.append(roc_auc_score(y[ix],p[ix]));specificity.append(np.mean(p[ix][~y[ix]]<.5))
    return {'status':'descriptive_complete_sensor_precision','n':48,'auroc':a['auroc'],
        'auroc_stratified_bootstrap_95':[float(x) for x in np.quantile(auc,[.025,.975])],
        'specificity_at_0_5':a['specificity_at_0_5'],
        'specificity_stratified_bootstrap_95':[float(x) for x in np.quantile(specificity,[.025,.975])],
        'audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest(),
        'limitations':'Exploratory public development sample; class-stratified bootstrap does not certify OOD or deployment calibration.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--measurements',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    a.output.write_text(json.dumps(compute(a.manifest,a.measurements,a.audit),indent=2)+'\n')
