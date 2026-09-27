#!/usr/bin/env python3
"""Audit common-pool branch evidence without inference or modifying labels."""
import argparse
import ast
import json
from pathlib import Path
import sys
import audit_mechanism as m
from prospective_replay import checked_task_contracts
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jev_control.branch_selectors import CandidateView,choose,local_prompt,parse_local_choice


class BranchLedger:
    def __init__(self,ledger,context,decision_time):
        self.ledger,self.context,self.decision_time=ledger,context,decision_time
    def take(self,ignored,*args,**kwargs):
        m.require(self.ledger.events[self.ledger.cursor]['started_unix']>=self.decision_time,'Outcome generated before selection')
        return self.ledger.take(self.context,*args,**kwargs)


def audit(run,tokenizer_dir):
    manifest=m.read_json(run/'manifest.json');summary=m.read_json(run/'summary.json')
    m.require(manifest['status']==summary['status']=='complete','Run incomplete')
    constants,tasks=checked_task_contracts(run,manifest)
    config=manifest['config']
    v3=config.get('protocol')=='common-pool-branch-replication-v3-development-only'
    config_name='branch_replication_v3.json' if v3 else 'branch_qualification_v2.json'
    m.require(config==m.read_json(run/'source/configs'/config_name),'Config mismatch')
    fixed={'protocol':'common-pool-branch-qualification-v2-development-only','problems':config['problems'],'repeats':2,'task_seed':791027,
        'budget':2048,'checkpoint_target':256,'checkpoint_cap':384,'final_reserve':128,'candidate_count':3,'candidate_tokens':128,'judge_tokens':128,'jev':False}
    if v3: fixed.update(protocol='common-pool-branch-replication-v3-development-only',problems=24,repeats=4,task_seed=891027,jev=True)
    m.require(all(config[k]==v for k,v in fixed.items()),'Unsupported settings')
    # The prompt/schema implementation is reviewed current code, never arbitrary source execution.
    name='src/jev_control/branch_selectors.py'
    m.require(m.digest((ROOT/name).read_bytes())==manifest['source_sha256'][name],'Unsupported selector/prompt implementation')
    if v3:
        name='src/jev_control/branch_jev.py'
        m.require(m.digest((ROOT/name).read_bytes())==manifest['source_sha256'][name],'Unsupported Jev prompt implementation')
    identity=manifest['model_identity']
    for name,digest in identity['file_sha256'].items():
        m.safe_relative(name)
        if name not in identity['weight_files_sha256']:
            m.require(m.digest((tokenizer_dir/name).read_bytes())==digest,'Tokenizer metadata mismatch')
    quant=manifest['actual_quantization_config']
    m.require(quant.get('bits')==4 and quant.get('group_size')==64,'Quantization mismatch')
    tokenizer=m.LocalTokenizer(tokenizer_dir)
    from transformers import AutoTokenizer
    chat_tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    chat=lambda text:list(chat_tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_dict=False))
    suffix=m.literal(ast.parse((run/'source/scripts/run_development.py').read_text()),'PROMPT_SUFFIX')
    schedule=m.read_json(run/'schedule.json')
    m.require(len(schedule)==config['problems'] and m.digest((run/'schedule.json').read_bytes())==manifest['schedule_sha256'],'Schedule mismatch')
    cps=m.indexed(m.read_jsonl(run/'checkpoints.jsonl'),'checkpoints');skips=m.indexed(m.read_jsonl(run/'skipped.jsonl'),'skipped')
    decisions=m.indexed(m.read_jsonl(run/'decisions.jsonl'),'decisions');local=m.indexed(m.read_jsonl(run/'local_judge.jsonl'),'local')
    m.require(not(set(cps)&set(skips)) and set(cps)==set(decisions)==set(local),'State/selection inventory mismatch')
    hosted=m.indexed(m.read_jsonl(run/'jev_judge.jsonl'),'jev') if v3 else {}
    if v3: m.require(set(hosted)==set(cps),'Jev inventory mismatch')
    candidates={};outcomes={}
    for row in m.read_jsonl(run/'candidates.jsonl'):
        key=(row['problem_id'],row['candidate']);m.require(key not in candidates,'Duplicate candidate');candidates[key]=row
    for row in m.read_jsonl(run/'outcomes.jsonl'):
        key=(row['problem_id'],row['repeat'],row['candidate']);m.require(key not in outcomes,'Duplicate outcome');outcomes[key]=row
    ledger=m.Ledger(run,tokenizer);seen_candidates=set();seen_outcomes=set()
    for i,item in enumerate(schedule):
        task=tasks['make_task'](i,config['task_seed']);task['prompt']+=suffix;pid=task['id'];base=config['task_seed']+i*10000
        m.require(item=={'index':i,'task':task,'initial_seed':config['task_seed']+i},'Task/seed mismatch')
        m.require(pid in cps or pid in skips,'Missing enrollment')
        cp=cps.get(pid,skips.get(pid));prompt=chat(task['prompt'])
        m.require(cp['prompt_ids']==prompt,'Initial prompt mismatch')
        initial=ledger.take({'phase':'initial','problem_id':pid,'index':i},prompt,384,item['initial_seed'],checkpoint=True,row_call=cp['initial'])
        m.online_boundary(initial,256,tokenizer)
        if pid in skips:
            m.require(initial['finish_reason']!='checkpoint','Eligible state skipped')
            m.require(cp['initial_outcome']==tasks['verify'](task,initial['text']),'Skipped outcome disagreement')
            expected=('completed_before_checkpoint' if initial['finish_reason']=='stop' else 'timeout_before_checkpoint' if initial['finish_reason']=='timeout' else 'answer_phase_reached_before_checkpoint' if 'FINAL:' in initial['text'] else 'no_boundary_before_cap')
            m.require(cp['reason']==expected,'Skip reason mismatch');continue
        m.require(initial['finish_reason']=='checkpoint','Ineligible state intervened on')
        m.checkpoint({**cp,'index':i},item,initial,tokenizer)
        pool=[]
        for j in range(3):
            key=(pid,j);m.require(key in candidates,'Missing candidate');seen_candidates.add(key)
            g=ledger.take({'phase':'candidate','problem_id':pid,'candidate':j},prompt+cp['retained_ids'],128,base+100+j,row_call=candidates[key]['generation'])
            pool.append(g)
        views=[CandidateView(g['text'],g['mean_logprob'],g['mean_entropy']) for g in pool]
        choices={};failures={}
        for method in ('uniform','likelihood','entropy_reduction'):
            try:choices[method]=choose(views,method,base+500)
            except ValueError:choices[method]=0;failures[method]='no_usable_score_fallback_first'
        expected_prompt=chat(local_prompt(task['prompt'],tokenizer.decode(cp['retained_ids']),views))
        m.require(local[pid]['prompt_ids']==expected_prompt,'Local prompt mismatch')
        local_start=ledger.events[ledger.cursor]['started_unix']
        lg=ledger.take({'phase':'local_selector','problem_id':pid,'temperature':0},expected_prompt,128,base+600,row_call=local[pid]['generation'])
        parsed=None
        try:
            if lg['finish_reason']!='timeout':parsed=parse_local_choice(lg['text'],3)
        except ValueError:pass
        m.require(local[pid]['choice']==parsed and local[pid]['error']==(None if parsed is not None else 'invalid_local_choice'),'Local parsing mismatch')
        choices['local_semantic']=parsed if parsed is not None else choices['likelihood']
        if parsed is None:failures['local_semantic']='invalid_choice_fallback_likelihood'
        if v3:
            from jev_control.branch_jev import request
            from jev_control.jev import canonical, MODEL, PRICE_PER_MILLION, RESERVE_USD
            jr=hosted[pid];state,questions=request(task['prompt'],tokenizer.decode(cp['retained_ids']),views)
            m.require(jr['request']=={'state':state,'questions':questions},'Jev information mismatch')
            if jr['error'] is None:
                result=jr['result'];response=result['response'];answer=response['answers']['candidate']
                m.require(response['model']==MODEL and answer['type']=='choice','Jev model/type mismatch')
                probs=answer['probabilities']
                m.require(set(probs)=={'0','1','2'} and all(type(v) in (int,float) and 0<=v<=1 for v in probs.values()),'Invalid Jev distribution')
                m.require(abs(sum(probs.values())-1.)<=1e-5,'Jev normalization')
                m.require(answer['choice'] in probs and probs[answer['choice']]>=max(probs.values())-1e-8,'Invalid Jev choice')
                m.require(jr['choice']==int(answer['choice']),'Jev parsing mismatch')
                payload={'model':MODEL,'state':state,'questions':questions}
                m.require(result['request_sha256']==m.digest(canonical(payload).encode()),'Jev request digest mismatch')
                m.require(jr['input_tokens']==response['usage']['input_tokens'],'Jev usage mismatch')
                m.equal_number(jr['accounted_usd'],jr['input_tokens']*PRICE_PER_MILLION/1e6,'Jev cost')
                choices['jev_semantic']=jr['choice']
            else:
                m.require(jr['choice'] is None,'Failed Jev choice used')
                m.equal_number(jr['accounted_usd'],RESERVE_USD,'Jev failure reserve')
                choices['jev_semantic']=choices['likelihood'];failures['jev_semantic']='invalid_choice_fallback_likelihood'
            m.require(jr['recorded_unix']>=local[pid]['recorded_unix'],'Jev chronology mismatch')
            m.require(decisions[pid]['recorded_unix']>=jr['recorded_unix'],'Decision before Jev')
        decision=decisions[pid];m.require(decision['choices']==choices and decision['failures']==failures,'Selector decision mismatch')
        m.require(decision['recorded_unix']>=local[pid]['recorded_unix']>=local_start,'Selection chronology mismatch')
        totals={'pool_generated_tokens':sum(g['generated_tokens'] for g in pool),'pool_prompt_tokens':sum(g['prompt_tokens'] for g in pool),
                'pool_service_seconds':sum(g['elapsed_seconds'] for g in pool)}
        for key,value in totals.items():m.equal_number(decision[key],value,key)
        for repeat in range(config['repeats']):
            seed=base+1000+repeat*100
            for j in (None,0,1,2):
                key=(pid,repeat,j);m.require(key in outcomes,'Missing outcome');seen_outcomes.add(key);row=outcomes[key]
                m.require(row['seed']==seed and row['action']=='continue','Outcome seed/action mismatch')
                is_branch=j is not None
                expected_selectors=[name for name,value in choices.items() if value==j] if is_branch else ['continue']
                m.require(row['selectors']==expected_selectors,'Selected-outcome mapping mismatch')
                selected=pool[j] if is_branch else None
                context={'phase':'shadow_continuation' if is_branch else 'baseline','problem_id':pid,'repeat':repeat}
                if is_branch:context['candidate']=j
                retained=cp['retained_ids']+(selected['token_ids'] if is_branch else [])
                pool_charge=totals['pool_generated_tokens'] if is_branch else 0
                if is_branch and selected['finish_reason'] in ('stop','timeout'):
                    m.require(row['calls']==[] and row['overhead']==[],'Ended candidate resumed')
                    m.require(row['text']==tokenizer.decode(retained),'Ended candidate text changed')
                    for name in ('generated_tokens','prompt_tokens_processed','elapsed_seconds'):m.equal_number(row[name],0,name)
                else:
                    adapted={**row,'index':i,'family':task['family'],'checkpoint_sha256':cp['sha256'],'shared_prefix_generated_tokens':initial['generated_tokens']}
                    m.audit_rollout(adapted,{**item,'scheduled':{'action':'continue','repeat':repeat,'seed':seed}},
                        {**cp,'retained_ids':retained},{**config,'budget':config['budget']-pool_charge},tokenizer,constants,
                        BranchLedger(ledger,context,decision['recorded_unix']))
                m.require(tasks['verify'](task,row['text'])==row['outcome'],'Outcome disagreement')
                expected={'episode_generated_tokens':initial['generated_tokens']+pool_charge+row['generated_tokens'],
                    'episode_prompt_tokens':initial['prompt_tokens']+(totals['pool_prompt_tokens'] if is_branch else 0)+row['prompt_tokens_processed'],
                    'episode_service_seconds':initial['elapsed_seconds']+(totals['pool_service_seconds'] if is_branch else 0)+row['elapsed_seconds'],
                    'selector_generated_tokens':lg['generated_tokens'] if is_branch else 0,
                    'selector_prompt_tokens':lg['prompt_tokens'] if is_branch else 0,
                    'selector_service_seconds':lg['elapsed_seconds'] if is_branch else 0.}
                for name,value in expected.items():m.equal_number(row[name],value,name)
                m.require(row['episode_generated_tokens']<=2048,'Episode generator budget exceeded')
                if is_branch and v3:
                    for field,source in [('jev_accounted_usd','accounted_usd'),('jev_acquisition_seconds','acquisition_seconds')]: m.equal_number(row[field],hosted[pid][source],field)
                    m.require(row['jev_input_tokens']==hosted[pid]['input_tokens'],'Jev token mismatch')
                if is_branch:
                    m.require(row['pool_generated_tokens']==pool_charge and row['selected_candidate_generated_tokens']==selected['generated_tokens'],'Pool cost mismatch')
                m.require(row['recorded_unix']>=decision['recorded_unix'],'Outcome recorded before decision')
    m.require(seen_candidates==set(candidates) and seen_outcomes==set(outcomes),'Extra records')
    m.require(set(cps)|set(skips)=={i['task']['id'] for i in schedule},'Extra enrollment')
    m.require(ledger.cursor==len(ledger.events),'Unaccounted calls')
    m.require(summary['completed_problems']==config['problems'] and summary['eligible_problems']==len(cps) and summary['outcomes']==len(outcomes),'Summary mismatch')
    return {'status':'passed_branch_qualification_audit','ready_for_analysis':True,'problems':config['problems'],'eligible_problems':len(cps),
        'outcomes':len(outcomes),'candidates':len(candidates),'durable_calls':len(ledger.events),'outcome_disagreements':0,
        'input_sha256':{p.name:m.digest(p.read_bytes()) for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')},
        'audit_source_sha256':m.digest(Path(__file__).read_bytes()),
        'limitations':['Recorded-contract audit, not fresh inference replay or weight rehash.',
            'Acquisition costs apply only to local_semantic; generator allowance is not total compute matching.',
            'Entropy selector is a published-method component adaptation, not full GUARD.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    r=audit(a.run,a.tokenizer);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
