#!/usr/bin/env python3
"""Fresh local-only repair qualification; no policy fitting, Jev or test data."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.repair_actions import ACTIONS, prepare
from jev_control.mlx_backend import MLXBackend
from jev_control.tasks import make_task, verify
from run_development import PROMPT_SUFFIX, boundary_now
from run_mechanism import LoggedBackend, checkpoint_from_initial, model_identity, write_json, append_jsonl, file_sha256
from run_screen import rollout
CONFIG=ROOT/'configs/action_qualification_v2.json'


def run(args):
    config=json.loads(CONFIG.read_text())
    if tuple(config['actions'])!=ACTIONS: raise ValueError('Action contract mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();manifest={'config':config,'status':'initializing','started_unix':time.time()}
    source={}
    for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src/jev_control').glob('*.py'))+[CONFIG,ROOT/'requirements.lock.txt']:
        relative=p.relative_to(ROOT);target=args.output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());source[str(relative)]=file_sha256(target)
    manifest['source_sha256']=source
    manifest['package_versions']={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','transformers')}
    schedule=[]
    for i in range(config['problems']):
        task=make_task(i,config['task_seed']);task['prompt']+=PROMPT_SUFFIX
        arms=[{'action':a,'repeat':r,'seed':config['task_seed']+i*10000+r*100} for a in ACTIONS for r in range(config['repeats'])]
        random.Random(config['task_seed']+i).shuffle(arms)
        schedule.append({'index':i,'task':task,'initial_seed':config['task_seed']+i,'arms':arms})
    write_json(args.output/'schedule.json',schedule);manifest['schedule_sha256']=file_sha256(args.output/'schedule.json')
    write_json(args.output/'manifest.json',manifest)
    for name in ('checkpoints.jsonl','outcomes.jsonl','generation_started.jsonl','generation_events.jsonl','skipped.jsonl'):
        (args.output/name).touch()
    backend=None;rows=[];skipped=[];status='failed'
    try:
        manifest['model_identity']=model_identity(args.model)
        raw=MLXBackend(args.model,temperature=config['temperature'],top_p=config['top_p'],quantization_bits=4)
        manifest['actual_quantization_config']=raw.quantization_config
        backend=LoggedBackend(raw,args.output,start+args.wall_seconds)
        manifest['status']='running';write_json(args.output/'manifest.json',manifest)
        for item in schedule:
            task=item['task'];pid=task['id']
            backend.context={'phase':'initial','problem_id':pid,'index':item['index']}
            prompt=backend.encode_chat([{'role':'user','content':task['prompt']}])
            initial=backend.generate(prompt,config['checkpoint_cap'],item['initial_seed'],
                stop_when=lambda ids:boundary_now(backend,ids,config['checkpoint_target']))
            cp,reason=checkpoint_from_initial(backend,task,prompt,initial,config['checkpoint_target'],config['checkpoint_cap'])
            if reason:
                record={'problem_id':pid,'reason':reason,'initial':initial.to_dict(),'prompt_ids':prompt,'initial_outcome':verify(task,initial.text)}
                skipped.append(record);append_jsonl(args.output/'skipped.jsonl',record)
                continue
            append_jsonl(args.output/'checkpoints.jsonl',cp)
            for arm in item['arms']:
                prepared,change=prepare(backend,cp['retained_ids'],arm['action'])
                backend.context={'phase':'continuation','problem_id':pid,**arm}
                row=rollout(backend,task,prompt,prepared,'continue',arm['seed'],config['budget']-initial.generated_tokens,config['final_reserve'])
                row.update(arm);row.update({'problem_id':pid,'family':task['family'],'preparation':change,
                    'prepared_ids':prepared,'checkpoint_sha256':cp['sha256'],'shared_prefix_generated_tokens':initial.generated_tokens})
                if row['generated_tokens']+initial.generated_tokens>config['budget']: raise ValueError('Budget violation')
                append_jsonl(args.output/'outcomes.jsonl',row);rows.append(row)
                print(json.dumps({'problem':item['index'],'action':arm['action'],'repeat':arm['repeat'],'success':row['outcome']['success']}),flush=True)
        status='complete'
    except BaseException as exc:
        manifest['error_type']=type(exc).__name__;raise
    finally:
        manifest.update(status=status,finished_unix=time.time());write_json(args.output/'manifest.json',manifest)
        write_json(args.output/'summary.json',{'status':status,'planned_problems':config['problems'],'skipped_problems':len(skipped),
            'recorded_outcomes':len(rows),'planned_max_outcomes':config['problems']*len(ACTIONS)*config['repeats'],
            'wall_seconds':time.monotonic()-start,'analysis_status':'Requires independent audit before interpretation'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,default=7200);a=p.parse_args()
    if not math.isfinite(a.wall_seconds) or a.wall_seconds<=0:p.error('Positive wall limit required')
    run(a)
