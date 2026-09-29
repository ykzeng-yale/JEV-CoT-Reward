"""Independent source, arithmetic, exact-prefix, call and outcome audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import audit_mechanism as m
from jev_control.tasks import verify
PAT=re.compile(r'(?<!\d)(\d{1,3})\s*\+\s*(\d{1,3})\s*=\s*(\d{1,3})(?!\d)')
FINAL='\n\nI must now provide the required final answer without further explanation.\nFINAL: '

def audit(run,tokenizer_dir):
    summary=m.read_json(run/'summary.json');manifest=m.read_json(run/'manifest.json')
    m.require(summary['status']==manifest['status']=='complete' and summary['outcomes']==50,'Incomplete study')
    for name,h in manifest['source_sha256'].items():
        m.safe_relative(name);m.require(m.digest((run/'source'/name).read_bytes())==h,'Frozen source changed')
    schedule=m.read_json(run/'schedule.json')
    m.require(len(schedule)==25 and m.digest((run/'schedule.json').read_bytes())==manifest['schedule_sha256'],'Schedule mismatch')
    tokenizer=m.LocalTokenizer(tokenizer_dir);ledger=m.Ledger(run,tokenizer)
    rows=m.read_jsonl(run/'outcomes.jsonl');m.require(len(rows)==50,'Outcome count mismatch')
    lookup={(r['problem_id'],r['repeat']):r for r in rows};m.require(len(lookup)==50,'Duplicate outcome')
    observed=set();timeout=0
    for item in schedule:
        source=ROOT/item['source_run'];cpfile=source/'checkpoints.jsonl';outfile=source/'outcomes.jsonl'
        m.require(m.digest(cpfile.read_bytes())==item['source_checkpoint_sha256'] and m.digest(outfile.read_bytes())==item['source_outcomes_sha256'],'Source data changed')
        cps={c['problem_id']:c for c in m.read_jsonl(cpfile)};cp=cps[item['problem_id']]
        m.require(cp['task']==item['task'] and cp['prompt_ids']==item['prompt_ids'] and cp['retained_ids']==item['retained_ids'],'Source checkpoint mismatch')
        match=PAT.search(cp['state']['latest_segment']);m.require(match is not None,'No explicit addition')
        a,b,c=map(int,match.groups());m.require([a,b,c]==item['claim'] and a+b!=c,'Not an exact natural error')
        instr=f'\n\nA checked calculation in the previous reasoning is wrong: {a}+{b}={c}. The correct sum is {a+b}. Reconsider only conclusions depending on that calculation, then continue.\n'
        m.require(item['instruction']==instr and item['initial_generated_tokens']==cp['initial']['generated_tokens'],'Incorrect feedback or cost')
        old=[r for r in m.read_jsonl(outfile) if r['problem_id']==item['problem_id'] and r['action']=='sham' and r['repeat']<2]
        m.require(len(old)==2,'Missing source sham');old={r['repeat']:r for r in old}
        prepared=item['retained_ids']+tokenizer.encode(instr)
        for arm in item['arms']:
            rep,seed=arm['repeat'],arm['seed'];key=(item['problem_id'],rep)
            m.require(key not in observed and rep in (0,1) and old[rep]['seed']==seed and old[rep]['outcome']['success']==arm['sham_success'],'Comparator mismatch')
            observed.add(key);row=lookup[key]
            m.require(row['seed']==seed and row['prepared_ids']==prepared and row['inserted_tokens']==len(tokenizer.encode(instr)),'Intervention mismatch')
            m.require(row['shared_prefix_generated_tokens']==item['initial_generated_tokens'],'Shared prefix cost mismatch')
            calls=row['calls'];remaining=2048-item['initial_generated_tokens'];reserve=128
            assistant=list(prepared);spent=0;position=0;overhead=[];context={'phase':'targeted_correction','problem_id':item['problem_id'],'repeat':rep}
            def take(cap,call_seed):
                nonlocal position,spent,assistant
                m.require(position<len(calls),'Missing recorded call')
                g=ledger.take(context,item['prompt_ids']+assistant,cap,call_seed,row_call=calls[position])
                position+=1;spent+=g['generated_tokens'];assistant+=g['token_ids'];return g
            if remaining-reserve>0:
                g=take(remaining-reserve,seed+200);done=g['finish_reason'] in ('stop','timeout')
            else:done=False
            if not done and remaining-spent>0:
                if 'FINAL:' not in tokenizer.decode(assistant):
                    ids=tokenizer.encode(FINAL);assistant+=ids;overhead.append({'kind':'finalization_instruction','tokens':len(ids)})
                g=take(remaining-spent,seed+300)
            m.require(position==len(calls) and spent==row['generated_tokens'] and spent<=remaining,'Call/cost mismatch')
            m.require(row['text']==tokenizer.decode(assistant) and row['overhead']==overhead,'Final text/prompt mismatch')
            m.require(row['prompt_tokens_processed']==sum(g['prompt_tokens'] for g in calls),'Prefill mismatch')
            m.require(row['outcome']==verify(item['task'],row['text']),'Terminal outcome mismatch')
            timeout+=any(g['finish_reason']=='timeout' for g in calls)
    m.require(observed==set(lookup) and ledger.cursor==len(ledger.events),'Extra outcomes/calls')
    return {'status':'passed_targeted_arithmetic_audit','eligible_checkpoints':25,'outcomes':50,'calls':len(ledger.events),'timeout_episodes':timeout,
        'outcome_disagreements':0,'input_sha256':{p.name:m.digest(p.read_bytes()) for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')},
        'audit_source_sha256':m.digest(Path(__file__).read_bytes()),
        'scope':'Matched prior sham comparator, nonconcurrent development continuation; exact arithmetic feedback only, not Jev-controlled inference.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refuse overwrite')
    a.output.write_text(json.dumps(audit(a.run,a.tokenizer),indent=2)+'\n')
