"""Freeze all natural explicit addition claims in two audited development runs."""
import hashlib
import json
import random
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[ROOT/'runs/action-replication-v3-20260928/checkpoints.jsonl',ROOT/'runs/sham-replication-v4-20260928/checkpoints.jsonl']
PAT=re.compile(r'(?<!\d)(\d{1,3})\s*\+\s*(\d{1,3})\s*=\s*(\d{1,3})(?!\d)')

def build():
    rows=[]
    for src in SOURCES:
        contents=src.read_bytes()
        for line in contents.decode().splitlines():
            cp=json.loads(line);state=cp['state'];match=PAT.search(state['latest_segment'])
            if match:
                a,b,c=map(int,match.groups())
                rows.append({'problem_id':cp['problem_id'],'state':state,'claim':[a,b,c],
                    'label':bool(a+b==c),'source_sha256':hashlib.sha256(contents).hexdigest()})
    if len(rows)!=48:raise ValueError(f'Expected48 natural claims,got{len(rows)}')
    random.Random(20260929).shuffle(rows)
    return {'protocol':'jev-natural-addition-v1-development-only','model':'jev-1.13.0','count':48,
        'rows':rows,'question':{'type':'noul','instructions':'Is the first explicit addition equality of the form number + number = number in latest_segment arithmetically true? Judge only that equality, even if the broader plan is tentative or another claim is wrong.'},
        'cumulative_stage_cap_usd':1.0,
        'purpose':'Natural-prefix arithmetic validity discrimination; no intervention outcomes or controller benefit claim.'}

if __name__=='__main__':
    p=ROOT/'configs/jev_natural_addition_v1.json'
    if p.exists():raise SystemExit('Refuse overwrite')
    p.write_text(json.dumps(build(),ensure_ascii=False,indent=2)+'\n')
