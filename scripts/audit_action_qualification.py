#!/usr/bin/env python3
"""Independent token/treatment/outcome audit for completed v2 repair qualification."""
import argparse
import ast
import json
from pathlib import Path
import random
import audit_mechanism as m
from prospective_replay import checked_task_contracts


def reconstruct(tokenizer, retained, action, instructions):
    """Reconstruct treatment from its declarative contract, not runner prepare()."""
    kept=len(retained);instruction=''
    if action=='suffix_repair':
        boundaries=[j for j in range(max(0,len(retained)-128)+1,len(retained))
                    if tokenizer.decode(retained[:j]).endswith('\n\n')]
        kept=max(boundaries) if boundaries else max(0,len(retained)-128)
        instruction=instructions['RECHECK']
    elif action=='segment_repair':
        end=len(tokenizer.decode(retained).rstrip())
        boundaries=[j for j in range(1,len(retained)) if
            len(tokenizer.decode(retained[:j]))<=end and tokenizer.decode(retained[:j]).endswith('\n\n')]
        kept=max(boundaries,default=0);instruction=instructions['SEGMENT']
    elif action=='recheck':instruction=instructions['RECHECK']
    elif action=='sham':instruction=instructions['SHAM']
    else:m.require(action=='continue','Unknown action')
    injected=tokenizer.encode(instruction) if instruction else []
    return retained[:kept]+injected,{'action':action,'removed_tokens':len(retained)-kept,
        'instruction':instruction,'inserted_tokens':len(injected),'kept_tokens':kept,
        'note':'Removed generation remains charged; inserted tokens incur prefill cost.'}


class ActionLedger:
    def __init__(self,ledger,action): self.ledger,self.action=ledger,action
    def take(self,context,*args,**kwargs):
        context={k:v for k,v in context.items() if k!='index'}
        context['action']=self.action
        return self.ledger.take(context,*args,**kwargs)


def audit(run,tokenizer_dir):
    manifest=m.read_json(run/'manifest.json');summary=m.read_json(run/'summary.json')
    m.require(manifest['status']==summary['status']=='complete','Run not complete')
    constants,tasks=checked_task_contracts(run,manifest)
    config=manifest['config']
    v3=config['protocol']=='repair-action-replication-v3-development-only'
    config_name='action_replication_v3.json' if v3 else 'action_qualification_v2.json'
    m.require(config==m.read_json(run/'source/configs'/config_name),'Config changed')
    m.require(v3 or config['protocol']=='repair-action-qualification-v2-development-only','Wrong protocol')
    fixed={'problems':8,'repeats':2,'task_seed':691027,'budget':2048,'checkpoint_target':256,'checkpoint_cap':384,'final_reserve':128,'jev':False}
    if v3:fixed.update(problems=24,repeats=4,task_seed=1491028)
    m.require(all(config[k]==v for k,v in fixed.items()),'Unsupported settings')
    m.require(manifest['actual_quantization_config']['bits']==4,'Wrong quantization')
    identity=manifest['model_identity']
    for name,digest in identity['file_sha256'].items():
        m.safe_relative(name)
        if name not in identity['weight_files_sha256']:
            m.require(m.digest((tokenizer_dir/name).read_bytes())==digest,'Tokenizer metadata changed')
    tokenizer=m.LocalTokenizer(tokenizer_dir)
    from transformers import AutoTokenizer
    chat=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    treatment_tree=ast.parse((run/'source/src/jev_control/repair_actions.py').read_text())
    actions=m.literal(treatment_tree,'ACTIONS');m.require(list(actions)==config['actions'],'Actions changed')
    instructions={k:m.literal(treatment_tree,k) for k in ('RECHECK','SHAM','SEGMENT')}
    suffix=m.literal(ast.parse((run/'source/scripts/run_development.py').read_text()),'PROMPT_SUFFIX')
    schedule=m.read_json(run/'schedule.json');m.require(len(schedule)==config['problems'],'Enrollment changed')
    m.require(m.digest((run/'schedule.json').read_bytes())==manifest['schedule_sha256'],'Schedule changed')
    cps=m.indexed(m.read_jsonl(run/'checkpoints.jsonl'),'checkpoints')
    skipped=m.indexed(m.read_jsonl(run/'skipped.jsonl'),'skipped')
    m.require(not(set(cps)&set(skipped)),'Enrollment overlap')
    outcomes=m.read_jsonl(run/'outcomes.jsonl');lookup={}
    for row in outcomes:
        key=(row['problem_id'],row['action'],row['repeat'])
        m.require(key not in lookup,'Duplicate outcome');lookup[key]=row
    ledger=m.Ledger(run,tokenizer);consumed=set();initial_tokens=0
    for i,item in enumerate(schedule):
        task=tasks['make_task'](i,config['task_seed']);task['prompt']+=suffix
        m.require(item['index']==i and item['task']==task,'Task mismatch')
        m.require(item['initial_seed']==config['task_seed']+i,'Initial seed mismatch')
        arms=[{'action':a,'repeat':r,'seed':config['task_seed']+i*10000+r*100} for a in actions for r in range(config['repeats'])]
        random.Random(config['task_seed']+i).shuffle(arms)
        m.require(item['arms']==arms,'Arm schedule changed')
        pid=task['id'];m.require(pid in cps or pid in skipped,'Missing enrollment')
        record=cps.get(pid,skipped.get(pid))
        prompt=list(chat.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False))
        m.require(record['prompt_ids']==prompt,'Chat prompt mismatch')
        initial=ledger.take({'phase':'initial','problem_id':pid,'index':i},prompt,384,item['initial_seed'],checkpoint=True,row_call=record['initial'])
        initial_tokens+=initial['generated_tokens'];m.online_boundary(initial,256,tokenizer)
        if pid in skipped:
            m.require(initial['finish_reason']!='checkpoint','Skipped eligible state')
            reason=('completed_before_checkpoint' if initial['finish_reason']=='stop' else
                'timeout_before_checkpoint' if initial['finish_reason']=='timeout' else
                'answer_phase_reached_before_checkpoint' if 'FINAL:' in initial['text'] else 'no_boundary_before_cap')
            m.require(record['reason']==reason,'Skip reason mismatch')
            m.require(record['initial_outcome']==tasks['verify'](task,initial['text']),'Skipped label mismatch')
            continue
        cp=cps[pid];m.require(initial['finish_reason']=='checkpoint','Ineligible intervention')
        m.checkpoint({**cp,'index':i},item,initial,tokenizer)
        for arm in arms:
            key=(pid,arm['action'],arm['repeat']);m.require(key in lookup,'Missing arm');row=lookup[key];consumed.add(key)
            prepared,change=reconstruct(tokenizer,cp['retained_ids'],arm['action'],instructions)
            m.require(row['preparation']==change and row['prepared_ids']==prepared,'Treatment reconstruction mismatch')
            m.require(row['action']==arm['action'],'Action mismatch')
            adapted={**row,'action':'continue','index':i}
            scheduled={**item,'scheduled':{**arm,'action':'continue'}}
            m.audit_rollout(adapted,scheduled,{**cp,'retained_ids':prepared},config,tokenizer,constants,ActionLedger(ledger,arm['action']))
            m.require(tasks['verify'](task,row['text'])==row['outcome'],'Terminal outcome mismatch')
    m.require(set(lookup)==consumed,'Unexpected outcomes')
    m.require(set(cps)|set(skipped)=={s['task']['id'] for s in schedule},'Extra enrollment')
    m.require(ledger.cursor==len(ledger.events),'Unknown or unconsumed calls')
    m.require(summary['recorded_outcomes']==len(outcomes) and summary['skipped_problems']==len(skipped),'Summary counts mismatch')
    return {'status':'passed_action_qualification_audit','ready_for_analysis':True,
        'problems':config['problems'],'eligible_problems':len(cps),'outcomes':len(outcomes),'durable_calls':len(ledger.events),
        'outcome_disagreements':0,'initial_tokens':initial_tokens,
        'input_sha256':{p.name:m.digest(p.read_bytes()) for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')},
        'audit_source_sha256':m.digest(Path(__file__).read_bytes()),
        'scope':'Exact recorded treatment/token/call/outcome reconstruction; not fresh inference replay, model weight rehash or efficacy evidence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    result=audit(a.run,a.tokenizer);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
