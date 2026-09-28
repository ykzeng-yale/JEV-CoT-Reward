#!/usr/bin/env python3
"""Complete-run provenance, token chain, chronology and label audit; no inference."""
import argparse
import ast
from pathlib import Path
import json
import sys
import audit_mechanism as m
from prospective_replay import checked_task_contracts
from audit_guard_episode import audit_episode
ROOT=Path(__file__).resolve().parents[1]


class ContextLedger:
    def __init__(self,ledger,context):self.ledger,self.context=ledger,context
    def take(self,ignored,*args,**kwargs):return self.ledger.take(self.context,*args,**kwargs)


def bind_records(row,call_records,decision_records,event_starts):
    """Require durable copies and chronological decisions before further inference."""
    def strip(r):return {k:v for k,v in r.items() if k not in ('problem_id','policy','recorded_unix')}
    m.require([strip(r) for r in call_records]==row['calls'],'Durable call copies disagree')
    m.require([strip(r) for r in decision_records]==row['decisions'],'Durable decision copies disagree')
    m.require(len(call_records)==len(event_starts),'Call time inventory mismatch')
    for i,(r,start) in enumerate(zip(call_records,event_starts)):
        m.require(start<=r['recorded_unix']<=row['recorded_unix'],'Call chronology mismatch')
        if i+1<len(event_starts):m.require(r['recorded_unix']<=event_starts[i+1],'Call recorded after next inference')
    # Each decision follows its segment, and a triggered decision follows all three candidates.
    ci=0
    for decision in decision_records:
        m.require(ci<len(call_records) and call_records[ci]['context']=='segment','Decision lacks segment')
        ci+=1
        if decision['trigger']:ci+=3
        m.require(ci<=len(call_records),'Incomplete candidate pool')
        m.require(call_records[ci-1]['recorded_unix']<=decision['recorded_unix']<=row['recorded_unix'],'Decision before evidence')
        if ci<len(event_starts):m.require(decision['recorded_unix']<=event_starts[ci],'Continuation preceded decision')


def audit(run,tokenizer_dir):
    manifest=m.read_json(run/'manifest.json');summary=m.read_json(run/'summary.json')
    m.require(manifest['status']==summary['status']=='complete','Run incomplete')
    constants,tasks=checked_task_contracts(run,manifest);config=manifest['config']
    cuda=config.get('backend')=='cuda'
    config_file='cuda_guard_runtime_v1.json' if cuda else 'guard_runtime_v1.json'
    m.require(config==m.read_json(run/'source/configs'/config_file),'Configuration mismatch')
    fixed={'protocol':'segment-guard-runtime-v1-development-only','problems':4,'task_seed':991027,'budget':2048,
           'checkpoint_target':256,'checkpoint_cap':384,'final_reserve':128,'segment_tokens':64,'branch_tokens':100,
           'temperature':.7,'top_p':.9,'repeats':1,'jev':False,
           'policies':['continue','segmented_sham','guard_adaptation']}
    if cuda:fixed.update(protocol='cuda-segment-guard-runtime-v1-development-only',problems=2,task_seed=1091027)
    m.require(all(config[k]==v for k,v in fixed.items()),'Unsupported configuration')
    for name in ('src/jev_control/guard_adaptation.py','src/jev_control/guard_runtime.py','scripts/run_guard_qualification.py'):
        m.require(m.digest((ROOT/name).read_bytes())==manifest['source_sha256'][name],'Unreviewed runtime implementation')
    tokenizer=m.LocalTokenizer(tokenizer_dir)
    identity=manifest['model_identity']
    m.require(identity['weight_revision_sha256']==m.digest(json.dumps(identity['weight_files_sha256'],sort_keys=True).encode()),'Weight revision mismatch')
    for name,digest in identity['file_sha256'].items():
        m.safe_relative(name)
        if name not in identity['weight_files_sha256']:
            m.require(m.digest((tokenizer_dir/name).read_bytes())==digest,'Tokenizer metadata mismatch')
    quant=manifest['actual_quantization_config']
    if cuda:
        m.require(quant=={'bits':16,'mode':'unquantized_bfloat16'},'Precision mismatch')
        name='src/jev_control/cuda_backend.py'
        m.require(m.digest((ROOT/name).read_bytes())==manifest['source_sha256'][name],'Unreviewed CUDA adapter')
        pre=m.read_json(run/'cuda_preflight.json');calls=pre['calls']
        m.require(pre['passed'] is True and len(calls)==3 and calls[0]['token_ids']==calls[1]['token_ids']+calls[2]['token_ids'],'CUDA preflight mismatch')
    else:m.require(quant.get('bits')==4 and quant.get('group_size')==64,'Quantization mismatch')
    from transformers import AutoTokenizer
    chat=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    encode=lambda text:list(chat.encode(text,add_special_tokens=False))
    suffix=m.literal(ast.parse((run/'source/scripts/run_development.py').read_text()),'PROMPT_SUFFIX')
    schedule=m.read_json(run/'schedule.json')
    m.require(len(schedule)==config['problems'] and m.digest((run/'schedule.json').read_bytes())==manifest['schedule_sha256'],'Schedule mismatch')
    cps=m.indexed(m.read_jsonl(run/'checkpoints.jsonl'),'checkpoints');skips=m.indexed(m.read_jsonl(run/'skipped.jsonl'),'skipped')
    outcomes={};durable={};decisions={}
    for r in m.read_jsonl(run/'outcomes.jsonl'):
        k=(r['problem_id'],r['policy']);m.require(k not in outcomes,'Duplicate outcome');outcomes[k]=r
    for filename,dest in [('calls',durable),('decisions',decisions)]:
        for r in m.read_jsonl(run/(filename+'.jsonl')):dest.setdefault((r['problem_id'],r['policy']),[]).append(r)
    m.require(not(set(cps)&set(skips)),'Duplicate enrollment')
    ledger=m.Ledger(run,tokenizer);seen=set();triggers=0
    for i,item in enumerate(schedule):
        task=tasks['make_task'](i,config['task_seed']);task['prompt']+=suffix;pid=task['id']
        m.require(item=={'index':i,'task':task,'initial_seed':config['task_seed']+i},'Task identity mismatch')
        m.require(pid in cps or pid in skips,'Missing enrollment')
        cp=cps.get(pid,skips.get(pid))
        prompt=list(chat.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False))
        m.require(cp['prompt_ids']==prompt,'Initial prompt mismatch')
        initial=ledger.take({'phase':'initial','problem_id':pid,'index':i},prompt,384,item['initial_seed'],checkpoint=True,row_call=cp['initial'])
        m.online_boundary(initial,256,tokenizer)
        if pid in skips:
            expected=('completed_before_checkpoint' if initial['finish_reason']=='stop' else 'timeout_before_checkpoint' if initial['finish_reason']=='timeout' else 'answer_phase_reached_before_checkpoint' if 'FINAL:' in initial['text'] else 'no_boundary_before_cap')
            m.require(initial['finish_reason']!='checkpoint' and cp['reason']==expected,'Invalid skipped checkpoint')
            m.require(cp['initial_outcome']==tasks['verify'](task,initial['text']),'Skipped outcome mismatch');continue
        m.require(initial['finish_reason']=='checkpoint','Ineligible intervention')
        m.checkpoint({**cp,'index':i},item,initial,tokenizer)
        for policy in config['policies']:
            key=(pid,policy);m.require(key in outcomes,'Missing policy outcome');seen.add(key);r=outcomes[key]
            seed=config['task_seed']+i*10000
            m.require(r['repeat']==0 and r['seed']==seed,'Seed/repeat mismatch')
            context={'phase':'qualification','problem_id':pid,'policy':policy}
            if policy=='continue':
                enriched={**r,'index':i,'family':task['family'],'checkpoint_sha256':cp['sha256'],'shared_prefix_generated_tokens':initial['generated_tokens']}
                m.audit_rollout(enriched,{**item,'scheduled':{'action':'continue','repeat':0,'seed':seed}},cp,config,tokenizer,constants,ContextLedger(ledger,context))
                m.require(r['episode_generated_tokens']==initial['generated_tokens']+r['generated_tokens'],'Continue accounting mismatch')
            else:
                audit_episode({**r,'generated_tokens':r['episode_generated_tokens']},prompt,cp['retained_ids'],initial['generated_tokens'],encode,seed=seed,branching=policy=='guard_adaptation')
                starts=[]
                for c in r['calls']:
                    starts.append(ledger.events[ledger.cursor]['started_unix'])
                    ledger.take({**context,'temperature':c['temperature']},c['prefix_ids'],c['requested_tokens'],c['seed'],row_call=c['generation'])
                bind_records(r,durable.get(key,[]),decisions.get(key,[]),starts)
                m.require(r['text']==tokenizer.decode(r['retained_ids']),'Final text mismatch')
                triggers+=sum(d['trigger'] for d in r['decisions'])
            m.require(r['outcome']==tasks['verify'](task,r['text']),'Outcome label mismatch')
    expected_segmented={k for k in seen if k[1]!='continue'}
    m.require(set(durable)==expected_segmented and set(decisions)<=expected_segmented,'Extra durable records')
    m.require(seen==set(outcomes) and ledger.cursor==len(ledger.events),'Extra outcomes/calls')
    m.require(set(cps)|set(skips)=={s['task']['id'] for s in schedule},'Extra enrollment')
    m.require(summary['completed_problems']==config['problems'] and summary['eligible_problems']==len(cps) and summary['outcomes']==len(outcomes),'Summary mismatch')
    return {'status':'passed_guard_runtime_audit','ready_for_analysis':True,'eligible_problems':len(cps),'outcomes':len(outcomes),
            'durable_calls':len(ledger.events),'natural_triggers':triggers,'outcome_disagreements':0,
            'input_sha256':{p.name:m.digest(p.read_bytes()) for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')},
            'audit_source_sha256':m.digest(Path(__file__).read_bytes()),
            'scope':'Recorded runtime qualification, not efficacy or original GUARD reproduction; zero triggers does not qualify real-model branch execution.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    result=audit(a.run,a.tokenizer);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
