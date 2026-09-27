#!/usr/bin/env python3
"""Release audited terminal evidence; exclude hosted features and machine paths."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def export(run, audit_path, destination):
    audit=json.loads(audit_path.read_text())
    if not (audit.get('ready_for_analysis') or audit.get('ready_for_statistical_analysis')) or not audit.get('status','').startswith('passed'):
        raise ValueError('Passed audit required')
    hashes=audit.get('input_sha256',audit.get('provenance',{}).get('input_sha256',{}))
    if not {'outcomes.jsonl','schedule.json'} <= set(hashes): raise ValueError('Incomplete audit inventory')
    for name,digest in hashes.items():
        if Path(name).name!=name or sha(run/name)!=digest: raise ValueError('Stale audit')
    fields=('problem_id','family','index','repeat','action','seed','outcome','text',
            'generated_tokens','prompt_tokens_processed','elapsed_seconds','policies',
            'hypothetical_deployment_costs','collection_sharing','decision_record_sha256',
            'candidate','selectors','episode_generated_tokens','episode_prompt_tokens',
            'episode_service_seconds','selector_generated_tokens','selector_prompt_tokens',
            'selector_service_seconds','jev_accounted_usd','jev_input_tokens','jev_acquisition_seconds','acquisition_note','pool_generated_tokens','selected_candidate_generated_tokens')
    rows=[{k:r[k] for k in fields if k in r} for r in lines(run/'outcomes.jsonl')]
    tasks=[s['task'] for s in json.loads((run/'schedule.json').read_text())]
    payload={'tasks.json':json.dumps(tasks,indent=2)+'\n',
             'outcomes.jsonl':''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)}
    for content in payload.values():
        if re.search(r'apikey_[a-f0-9]{20,}|-----BEGIN .*PRIVATE KEY|/Users/|/home/|github_pat_',content):
            raise ValueError('Sensitive content in release')
    if destination.exists(): raise ValueError('Destination exists')
    destination.mkdir(parents=True)
    for name,content in payload.items(): (destination/name).write_text(content)
    manifest={'original_run':run.name,'original_audit_sha256':sha(audit_path),
              'original_input_sha256':hashes,
              'released_sha256':{n:sha(destination/n) for n in payload},
              'scope':'Terminal outcome reproduction only. No hosted responses, fitted controller, token-call replay or feature analysis.',
              'rows':len(rows),'problems':len(tasks)}
    (destination/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('audit',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();export(a.run,a.audit,a.destination)
