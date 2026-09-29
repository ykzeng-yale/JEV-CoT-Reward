"""Bounded Jev sensor measurement on a frozen paired arithmetic manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.jev import JevClient

def run(config,out):
    m=json.loads(config.read_text());assert m['n_pairs']==16 and m['model']=='jev-1.13.0'
    out.mkdir(parents=True,exist_ok=False)
    client=JevClient(stage_cap_usd=m['cumulative_stage_cap_usd'])
    started=time.time();rows=[]
    status={'status':'running','manifest_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_unix':started,
            'ledger_before':client.status(),'planned_requests':32}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        for i,p in enumerate(m['pairs']):
            for kind in p['order']:
                state=dict(p['state'])
                if kind==1:state['latest_segment']=p['altered_segment']
                result=client.evaluate(state,{'equality_true':m['question']})
                value=result['response']['answers']['equality_true']['noul']
                row={'pair':i,'kind':'correct' if kind==0 else 'corrupted','noul':value,
                     'request_sha256':result['request_sha256'],'input_cost_usd':result['input_cost_usd'],
                     'input_tokens':result['response']['usage']['input_tokens'],
                     'elapsed_seconds':result['elapsed_seconds'],'cache_hit':result['cache_hit']}
                with (out/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                rows.append(row)
        status['status']='complete'
    except Exception as exc:
        status['status']='failed';status['error_type']=type(exc).__name__
        raise
    finally:
        status.update(finished_unix=time.time(),completed_requests=len(rows),ledger_after=client.status())
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config,a.output)
