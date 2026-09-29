"""One new evidence-grounded correction arm on frozen natural-error checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.mlx_backend import MLXBackend
from jev_control.tasks import verify
from run_mechanism import LoggedBackend,model_identity,write_json,append_jsonl
from run_screen import rollout

SOURCES=[ROOT/'runs/action-replication-v3-20260928',ROOT/'runs/sham-replication-v4-20260928']
PAT=re.compile(r'(?<!\d)(\d{1,3})\s*\+\s*(\d{1,3})\s*=\s*(\d{1,3})(?!\d)')

def schedule():
    items=[]
    for run in SOURCES:
        cps={r['problem_id']:r for r in map(json.loads,(run/'checkpoints.jsonl').read_text().splitlines())}
        outcomes=[json.loads(x) for x in (run/'outcomes.jsonl').read_text().splitlines()]
        for pid,cp in cps.items():
            m=PAT.search(cp['state']['latest_segment'])
            if not m:continue
            a,b,c=map(int,m.groups())
            if a+b==c:continue
            old=[r for r in outcomes if r['problem_id']==pid and r['action']=='sham' and r['repeat']<2]
            if len(old)!=2:raise ValueError('Missing matched sham')
            instruction=f'\n\nA checked calculation in the previous reasoning is wrong: {a}+{b}={c}. The correct sum is {a+b}. Reconsider only conclusions depending on that calculation, then continue.\n'
            items.append({'source_run':str(run.relative_to(ROOT)),'source_checkpoint_sha256':hashlib.sha256((run/'checkpoints.jsonl').read_bytes()).hexdigest(),
                'source_outcomes_sha256':hashlib.sha256((run/'outcomes.jsonl').read_bytes()).hexdigest(),
                'problem_id':pid,'task':cp['task'],'prompt_ids':cp['prompt_ids'],'retained_ids':cp['retained_ids'],
                'initial_generated_tokens':cp['initial']['generated_tokens'],'claim':[a,b,c],
                'instruction':instruction,'arms':[{'repeat':r['repeat'],'seed':r['seed'],'sham_success':r['outcome']['success']} for r in sorted(old,key=lambda x:x['repeat'])]})
    if len(items)!=25:raise ValueError(f'Expected25 natural errors,got{len(items)}')
    return items

def run(model,out,wall_seconds):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    items=schedule();write_json(out/'schedule.json',items)
    source={}
    for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src/jev_control').glob('*.py'))+[ROOT/'docs/targeted_arithmetic_correction_v1_protocol.md']:
        dst=out/'source'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
        source[str(p.relative_to(ROOT))]=hashlib.sha256(dst.read_bytes()).hexdigest()
    manifest={'status':'initializing','schedule_sha256':hashlib.sha256((out/'schedule.json').read_bytes()).hexdigest(),
        'source_sha256':source,'started_unix':time.time(),'package_versions':{p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','transformers')}}
    write_json(out/'manifest.json',manifest)
    for name in ('generation_started','generation_events','outcomes'):(out/(name+'.jsonl')).touch()
    n=0;status='failed'
    try:
        manifest['model_identity']=model_identity(model)
        raw=MLXBackend(model,temperature=.7,top_p=.9,quantization_bits=4)
        manifest['actual_quantization_config']=raw.quantization_config
        backend=LoggedBackend(raw,out,start+wall_seconds)
        manifest['status']='running';write_json(out/'manifest.json',manifest)
        for item in items:
            inserted=backend.encode_text(item['instruction']);retained=item['retained_ids']+inserted
            for arm in item['arms']:
                backend.context={'phase':'targeted_correction','problem_id':item['problem_id'],'repeat':arm['repeat']}
                row=rollout(backend,item['task'],item['prompt_ids'],retained,'continue',arm['seed'],2048-item['initial_generated_tokens'],128)
                row.update(problem_id=item['problem_id'],repeat=arm['repeat'],seed=arm['seed'],
                    inserted_tokens=len(inserted),prepared_ids=retained,shared_prefix_generated_tokens=item['initial_generated_tokens'])
                append_jsonl(out/'outcomes.jsonl',row);n+=1
            print(json.dumps({'completed_pairs':n//2,'total_pairs':len(items)}),flush=True)
        status='complete'
    except BaseException as exc:
        manifest['error_type']=type(exc).__name__;raise
    finally:
        manifest.update(status=status,finished_unix=time.time());write_json(out/'manifest.json',manifest)
        write_json(out/'summary.json',{'status':status,'outcomes':n,'planned':50,'elapsed_seconds':time.monotonic()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-seconds',type=float,default=3600);a=p.parse_args();run(a.model,a.output,a.wall_seconds)
