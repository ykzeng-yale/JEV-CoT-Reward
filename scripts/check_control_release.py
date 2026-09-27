#!/usr/bin/env python3
"""Recompute released terminal labels/counts without models, hosted calls or pickle."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from jev_control.tasks import verify


def check(root):
    manifest=json.loads((root/'release_manifest.json').read_text())
    for name,digest in manifest['released_sha256'].items():
        if Path(name).name!=name or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Release changed')
    tasks={t['id']:t for t in json.loads((root/'tasks.json').read_text())}
    rows=[json.loads(s) for s in (root/'outcomes.jsonl').read_text().splitlines()]
    counts=defaultdict(lambda:[0,0]);seen=set();branch_rows=set()
    for r in rows:
        if verify(tasks[r['problem_id']],r['text']) != r['outcome']: raise ValueError('Outcome disagreement')
        if 'candidate' in r:
            identity=(r['problem_id'],r.get('repeat',0),r['candidate'])
            if identity in branch_rows: raise ValueError('Duplicate branch outcome')
            branch_rows.add(identity)
        groups=r.get('selectors',r.get('policies',[r['action']]))
        for group in groups:
            key=(r['problem_id'],group,r.get('repeat',0))
            if key in seen: raise ValueError('Duplicate episode')
            seen.add(key)
            counts[group][0]+=int(r['outcome']['success']);counts[group][1]+=1
    if len(rows)!=manifest['rows'] or len(tasks)!=manifest['problems']: raise ValueError('Inventory mismatch')
    return {'problems':len(tasks),'unique_outcomes':len(rows),'verified_counts':dict(counts),'outcome_disagreements':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path)
    print(json.dumps(check(p.parse_args().release),indent=2))
