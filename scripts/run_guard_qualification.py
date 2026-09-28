#!/usr/bin/env python3
"""Bounded runtime qualification: continue, segmented sham and GUARD adaptation."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.guard_adaptation import run as guard_run
from jev_control.guard_runtime import scoped_generate
from jev_control.mlx_backend import MLXBackend
from jev_control.prospective import durable_record
from jev_control.tasks import make_task, verify
from run_development import PROMPT_SUFFIX,boundary_now
from run_mechanism import LoggedBackend,checkpoint_from_initial,model_identity,write_json,file_sha256
from run_screen import rollout
CONFIG=ROOT/'configs/guard_runtime_v1.json'


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
    packages=('torch','numpy','transformers') if config.get('backend')=='cuda' else ('mlx','mlx-lm','numpy','transformers')
    manifest={'status':'initializing','config':config,'started_unix':time.time(),'source_sha256':source,
        'schedule_sha256':file_sha256(args.output/'schedule.json'),
        'package_versions':{p:importlib.metadata.version(p) for p in packages}}
    write_json(args.output/'manifest.json',manifest)
    names=('checkpoints','outcomes','generation_started','generation_events','skipped','calls','decisions')
    for name in names:(args.output/(name+'.jsonl')).touch()
    def record(kind,row):durable_record(args.output/(kind+'.jsonl'),{**row,'recorded_unix':time.time()})
    completed=eligible=outcomes=0;status='failed'
    try:
        manifest['model_identity']=model_identity(args.model)
        if config.get('backend')=='cuda':
            from jev_control.cuda_backend import CUDABackend
            raw=CUDABackend(args.model,temperature=config['temperature'],top_p=config['top_p'])
        else:
            raw=MLXBackend(args.model,temperature=config['temperature'],top_p=config['top_p'],quantization_bits=4)
        manifest['actual_quantization_config']=raw.quantization_config;manifest['status']='running'
        write_json(args.output/'manifest.json',manifest)
        if config.get('backend')=='cuda':
            original=raw._sampler;raw._sampler=0.
            check_prompt=raw.encode_chat([{'role':'user','content':'Explain Dijkstra shortest path algorithm step by step.'}])
            whole=raw.generate(check_prompt,32,9)
            first=raw.generate(check_prompt,16,9)
            rest=raw.generate(check_prompt+first.token_ids,16,9)
            raw._sampler=original
            preflight={'passed':whole.token_ids==first.token_ids+rest.token_ids,'calls':[g.to_dict() for g in (whole,first,rest)],
                       'scope':'Infrastructure work, separate from research episodes but included in total allocated GPU time.'}
            write_json(args.output/'cuda_preflight.json',preflight)
            if not preflight['passed']:raise ValueError('CUDA adapter greedy resume failed')
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
                for policy in config['policies']:
                    seed=config['task_seed']+index*10000
                    backend.context={'phase':'qualification','problem_id':task['id'],'policy':policy}
                    if policy=='continue':
                        row=rollout(backend,task,prompt,cp['retained_ids'],'continue',seed,config['budget']-initial.generated_tokens,config['final_reserve'])
                        row.update(episode_generated_tokens=initial.generated_tokens+row['generated_tokens'])
                    else:
                        def save(kind,row):record(kind,{'problem_id':task['id'],'policy':policy,**row})
                        result=guard_run(prompt,cp['retained_ids'],initial.generated_tokens,
                            lambda p,c,s,t:scoped_generate(backend,p,c,s,t),backend.encode_text,save,
                            budget=config['budget'],reserve=config['final_reserve'],segment_tokens=config['segment_tokens'],
                            branch_tokens=config['branch_tokens'],seed=seed,branching=policy in ('guard_adaptation','random_branch'),
                            random_rate=config['random_rate'] if policy=='random_branch' else None)
                        text=backend.decode(result['retained_ids'])
                        row={'text':text,'outcome':verify(task,text),'episode_generated_tokens':result['generated_tokens'],
                             'retained_ids':result['retained_ids'],'calls':result['calls'],'decisions':result['decisions']}
                    record('outcomes',{'problem_id':task['id'],'policy':policy,'repeat':0,'seed':seed,**row})
                    outcomes+=1
            completed+=1
            print(json.dumps({'completed_problems':completed,'eligible_problems':eligible,'outcomes':outcomes}),flush=True)
        status='complete'
    except BaseException as exc:
        manifest['error_type']=type(exc).__name__;raise
    finally:
        manifest.update(status=status,finished_unix=time.time());write_json(args.output/'manifest.json',manifest)
        write_json(args.output/'summary.json',{'status':status,'completed_problems':completed,'eligible_problems':eligible,
            'planned_problems':config['problems'],'outcomes':outcomes,'wall_seconds':time.monotonic()-start,
            'analysis_status':'No policy benefit or complete data claim until independent audit passes.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--wall-seconds',type=float,default=7200);a=p.parse_args()
    if not math.isfinite(a.wall_seconds) or a.wall_seconds<=0:p.error('Finite positive deadline required')
    run(a)
