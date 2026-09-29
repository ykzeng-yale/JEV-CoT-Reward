"""Freeze a paired diagnostic from public development prefixes; no API or outcomes."""
import hashlib
import json
import random
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[ROOT/'runs/action-replication-v3-20260928/checkpoints.jsonl',ROOT/'runs/sham-replication-v4-20260928/checkpoints.jsonl']
PAT=re.compile(r'(?<!\d)(\d{1,3})\s*\+\s*(\d{1,3})\s*=\s*(\d{1,3})(?!\d)')


def build():
    selected=[]
    for src in SOURCES:
        for line in src.read_text().splitlines():
            row=json.loads(line);state=row['state'];segment=state['latest_segment']
            for match in PAT.finditer(segment):
                a,b,c=map(int,match.groups())
                if a+b!=c:continue
                altered=segment[:match.start(3)]+str(c+1)+segment[match.end(3):]
                if altered==segment:raise ValueError('No perturbation')
                selected.append({'problem_id':row['problem_id'],'family':row['task']['family'],
                    'state':state,'altered_segment':altered,'claim':[a,b,c],
                    'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
                break
            if len(selected)>=16:break
        if len(selected)>=16:break
    if len(selected)!=16:raise ValueError('Insufficient valid natural claims')
    rng=random.Random(20260928)
    manifest=[]
    for item in selected:
        order=list(range(2));rng.shuffle(order)
        manifest.append({**item,'order':order})
    return {'protocol':'jev-paired-arithmetic-sensor-v1','n_pairs':16,'pairs':manifest,
        'question':{'type':'noul','instructions':'Is the first explicit addition equality of the form number + number = number in latest_segment arithmetically true? Judge only that equality, even if the broader plan is tentative or another claim is wrong.'},
        'model':'jev-1.13.0','cumulative_stage_cap_usd':1.0,
        'scope':'Off-policy matched arithmetic perturbation diagnostic; no intervention or downstream success claim.'}

if __name__=='__main__':
    p=ROOT/'configs/jev_arithmetic_pairs_v1.json'
    if p.exists():raise SystemExit('Refuse overwrite')
    p.write_text(json.dumps(build(),ensure_ascii=False,indent=2)+'\n')
