#!/usr/bin/env python3
"""Bounded local-only common-pool experiment, with gold-free pre-outcome selection."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.branch_experiment import collect_pool
from jev_control.mlx_backend import MLXBackend
from jev_control.prospective import durable_record
from jev_control.tasks import make_task, verify
from run_development import PROMPT_SUFFIX,boundary_now
from run_mechanism import LoggedBackend,checkpoint_from_initial,model_identity,write_json,file_sha256
from run_screen import rollout
CONFIG=ROOT/'configs/branch_qualification_v2.json'


def run(args):
    config_path=Path(getattr(args,'config',CONFIG)).resolve()
    config=json.loads(config_path.read_text());start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=False)
    source={}
    for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'src/jev_control').glob('*.py'))+[config_path,ROOT/'requirements.lock.txt']:
        relative=p.relative_to(ROOT);target=args.output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());source[str(relative)]=file_sha256(target)
    schedule=[]
    for index in range(config['problems']):
        task=make_task(index,config['task_seed']);task['prompt']+=PROMPT_SUFFIX
        schedule.append({'index':index,'task':task,'initial_seed':config['task_seed']+index})
    write_json(args.output/'schedule.json',schedule)
    manifest={'status':'initializing','config':config,'started_unix':time.time(),'source_sha256':source,
        'schedule_sha256':file_sha256(args.output/'schedule.json'),
        'package_versions':{p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','transformers')}}
    write_json(args.output/'manifest.json',manifest)
    jev_client=None
    if config.get('jev'):
        from jev_control.jev import JevClient
        jev_client=JevClient(stage_cap_usd=config['jev_cumulative_stage_cap_usd'])
        manifest['jev_ledger_before']=jev_client.status()
    names=('jev_judge','checkpoints','outcomes','generation_started','generation_events','skipped','candidates','decisions','local_judge')
    for name in names:(args.output/(name+'.jsonl')).touch()
    def record(kind,row):durable_record(args.output/(kind+'.jsonl'),{**row,'recorded_unix':time.time()})
    completed=eligible=outcomes=0;status='failed'
    try:
        manifest['model_identity']=model_identity(args.model)
        raw=MLXBackend(args.model,temperature=config['temperature'],top_p=config['top_p'],quantization_bits=4)
        manifest['actual_quantization_config']=raw.quantization_config;manifest['status']='running'
        write_json(args.output/'manifest.json',manifest)
        backend=LoggedBackend(raw,args.output,start+args.wall_seconds)
        for item in schedule:
            task=item['task'];index=item['index']
            backend.context={'phase':'initial','problem_id':task['id'],'index':index}
            prompt=backend.encode_chat([{'role':'user','content':task['prompt']}])
            initial=backend.generate(prompt,config['checkpoint_cap'],item['initial_seed'],
                stop_when=lambda ids:boundary_now(backend,ids,config['checkpoint_target']))
            cp,reason=checkpoint_from_initial(backend,task,prompt,initial,config['checkpoint_target'],config['checkpoint_cap'])
            if reason:
                record('skipped',{'problem_id':task['id'],'reason':reason,'initial':initial.to_dict(),
                    'prompt_ids':prompt,'initial_outcome':verify(task,initial.text)})
            else:
                record('checkpoints',cp);eligible+=1
                _,rows=collect_pool(backend,task,cp,config,index,record,rollout,jev_client=jev_client)
                outcomes+=len(rows)
            completed+=1
            print(json.dumps({'completed_problems':completed,'eligible_problems':eligible,'outcomes':outcomes}),flush=True)
        status='complete'
    except BaseException as exc:
        manifest['error_type']=type(exc).__name__;raise
    finally:
        if jev_client is not None: manifest['jev_ledger_after']=jev_client.status()
        manifest.update(status=status,finished_unix=time.time());write_json(args.output/'manifest.json',manifest)
        write_json(args.output/'summary.json',{'status':status,'completed_problems':completed,'eligible_problems':eligible,
            'planned_problems':config['problems'],'outcomes':outcomes,'wall_seconds':time.monotonic()-start,
            'analysis_status':'No policy benefit or complete data claim until independent audit passes.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--wall-seconds',type=float,default=7200);a=p.parse_args()
    if not math.isfinite(a.wall_seconds) or a.wall_seconds<=0:p.error('Finite positive deadline required')
    run(a)
