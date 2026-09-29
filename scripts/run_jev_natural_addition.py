"""Bounded Jev natural-prefix sensor study; stops on first API failure."""
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
    m=json.loads(config.read_text());assert m['count']==48
    out.mkdir(parents=True,exist_ok=False)
    client=JevClient(stage_cap_usd=m['cumulative_stage_cap_usd'])
    status={'status':'running','manifest_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_unix':time.time(),'ledger_before':client.status(),'planned_requests':48}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    done=0
    try:
        for i,item in enumerate(m['rows']):
            response=client.evaluate(item['state'],{'equality_true':m['question']})
            row={'index':i,'noul':response['response']['answers']['equality_true']['noul'],
                'request_sha256':response['request_sha256'],
                'input_cost_usd':response['input_cost_usd'],'cache_hit':response['cache_hit'],
                'input_tokens':response['response']['usage']['input_tokens'],
                'elapsed_seconds':response['elapsed_seconds']}
            with (out/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            done+=1
        status['status']='complete'
    except Exception as exc:
        status['status']='failed';status['error_type']=type(exc).__name__
        raise
    finally:
        status.update(completed_requests=done,finished_unix=time.time(),ledger_after=client.status())
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config,a.output)
