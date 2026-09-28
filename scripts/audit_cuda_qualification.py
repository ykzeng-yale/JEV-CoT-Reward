#!/usr/bin/env python3
"""Check retrieved CUDA qualification outputs against submission and accounting."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(run,accounting):
    def read(name):return json.loads((run/name).read_text())
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
    receipt=read('submission.json');manifest=read('outputs/manifest.json');summary=read('outputs/summary.json')
    resume=read('outputs/resume_check.json');long=read('outputs/long_context.json');batch=read('outputs/batch.json')
    records=[line.strip().split('|') for line in accounting.read_text().splitlines() if line.strip()]
    matching=[r for r in records if r[0]==receipt['job_id']]
    require(len(matching)==1 and matching[0][1:3]==['COMPLETED','0:0'],'Slurm job did not complete successfully')
    require(manifest['job_id']==receipt['job_id'],'Job ID mismatch')
    require(manifest['model']==receipt['model'] and manifest['revision']==receipt['revision'],'Model identity mismatch')
    require(manifest['script_sha256']==receipt['source_sha256']['qualify_cuda.py'],'Script digest mismatch')
    require(manifest['dtype']=='bfloat16' and manifest['quantization'] is None,'Precision mismatch')
    require(manifest['versions']['torch']=='2.9.1' and manifest['versions']['transformers']=='4.55.2' and manifest['cuda']=='12.8','Runtime mismatch')
    require(manifest['gpu']==summary['gpu'] and any(g in manifest['gpu'] for g in ('H100','H200','B200')),'Unexpected GPU')
    require(type(manifest['snapshot_bytes']) is int and 0<manifest['snapshot_bytes']<16*1024**3,'Model footprint violation')
    require(summary['status']=='passed' and resume['greedy_exact_prefix_resume'] is True,'Resume qualification failed')
    whole=resume['whole_ids'];resumed=resume['resumed_ids']
    require(whole==resumed and len(whole)==resume['generated_tokens']==64,'Saved token sequences disagree')
    require(all(type(t) is int and t>=0 for t in whole),'Invalid saved token IDs')
    require(long['input_tokens']==8192 and long['output_tokens']==64 and positive(long['seconds']),'Long-context probe incomplete')
    require(len(batch['outputs'])==4 and positive(batch['elapsed_seconds']),'Batch probe incomplete')
    for row in batch['outputs']:
        require(0<len(row['token_ids'])<=256 and all(type(t) is int and t>=0 for t in row['token_ids']),'Invalid batch tokens')
        require(isinstance(row['text'],str),'Missing batch text')
    emitted=sum(len(r['token_ids']) for r in batch['outputs'])
    require(batch['emitted_tokens']==emitted,'Batch token accounting mismatch')
    throughput=emitted/batch['elapsed_seconds']
    require(positive(summary['batch_tokens_per_second']) and math.isclose(throughput,summary['batch_tokens_per_second'],rel_tol=1e-9),'Throughput mismatch')
    require(positive(summary['peak_allocated_gib']),'Invalid GPU memory measurement')
    return {'status':'passed_cuda_qualification_record_audit','job_id':receipt['job_id'],'gpu':manifest['gpu'],
            'batch_tokens_per_second':throughput,'peak_allocated_gib':summary['peak_allocated_gib'],
            'input_sha256':{str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob('*.json')},
            'accounting_sha256':hashlib.sha256(accounting.read_bytes()).hexdigest(),
            'scope':'Saved infrastructure-output consistency, not fresh GPU replay, BF16/MLX equivalence or scientific efficacy.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--accounting',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    result=audit(a.run,a.accounting);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
