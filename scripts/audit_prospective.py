#!/usr/bin/env python3
"""Audit completed prospective records without generation or hosted requests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import audit_mechanism as m
import analyze_screen as screen
from jev_control.prospective_audit import assignment_audit
from local_judge import judge_prompt, parse_probabilities
from prospective_replay import checked_task_contracts, audit_selected_rollout
from run_prospective import load_frozen


def audit(run, tokenizer_dir):
    manifest=m.read_json(run/'manifest.json');summary=m.read_json(run/'summary.json')
    m.require(manifest['status']==summary['status']=='complete','Prospective run incomplete')
    config=manifest['config']
    m.require(config==m.read_json(run/'source/configs/prospective_diagnostic_v1.json'),'Config differs from launch')
    m.require(summary['completed_problems']==summary['planned_problems']==config['problems']==24,'Incomplete enrollment')
    constants,tasks=checked_task_contracts(run,manifest)
    identity=manifest['model_identity']
    for name,digest in identity['file_sha256'].items():
        m.safe_relative(name)
        if name not in identity['weight_files_sha256']:
            m.require(m.digest((tokenizer_dir/name).read_bytes())==digest,'Tokenizer metadata mismatch')
    tokenizer=m.LocalTokenizer(tokenizer_dir)
    from transformers import AutoTokenizer
    chat_tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir),local_files_only=True,trust_remote_code=False)
    chat=lambda text: list(chat_tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True))
    feature_tree=ast.parse((run/'source/src/jev_control/features.py').read_text())
    rubric_record=m.read_json(run/'rubric.json')
    m.require(rubric_record=={'schema':m.literal(feature_tree,'SCHEMA_VERSION'),
                            'questions':m.literal(feature_tree,'QUESTIONS')},'Rubric changed from frozen questions')
    quant=manifest['actual_quantization_config']
    m.require(quant.get('bits')==4 and quant.get('group_size')==64 and quant.get('mode','affine')=='affine','Unexpected loaded quantization')
    schedule=m.read_json(run/'schedule.json')
    m.require(m.digest((run/'schedule.json').read_bytes())==manifest['schedule_sha256'],'Schedule hash mismatch')
    checkpoints=m.read_jsonl(run/'checkpoints.jsonl');rows=m.read_jsonl(run/'outcomes.jsonl')
    decisions_bytes=(run/'decisions.jsonl').read_bytes()
    ledger=m.Ledger(run,tokenizer)
    assignment=assignment_audit(schedule,checkpoints,decisions_bytes,rows,ledger.events)
    controllers,frozen=load_frozen(run/'frozen_controllers')
    m.require(frozen['artifact_sha256']==manifest['controller_artifact_sha256'],'Controller identity mismatch')
    m.require(frozen['created_unix']<manifest['started_unix'],'Controller not frozen before launch')
    cps={c['problem_id']:c for c in checkpoints}
    decisions={d['problem_id']:d for d in map(json.loads,decisions_bytes.splitlines())}
    predictions=m.indexed(m.read_jsonl(run/'predictions.jsonl'),'predictions')
    local=m.indexed(m.read_jsonl(run/'local_judge.jsonl'),'local features')
    outcome_disagreements=[];predicted_count=0
    for index,item in enumerate(schedule):
        task=item['task'];pid=task['id'];cp=cps[pid]
        expected=tasks['make_task'](index,config['task_seed'])
        # Literal captured prompt suffix, inspected through the mechanism contract.
        tree=ast.parse((run/'source/scripts/run_development.py').read_text())
        expected['prompt']+=m.literal(tree,'PROMPT_SUFFIX')
        m.require(item['index']==index and task==expected,'Task schedule changed')
        m.require(pid not in frozen['training_problem_ids'],'Training/test overlap')
        m.require(item['initial_seed']==config['task_seed']+index and item['continuation_seed']==config['task_seed']+index*10000,'Seed schedule changed')
        m.require(cp['prompt_ids']==chat(task['prompt']),'Initial chat prompt differs from task')
        initial=ledger.take({'phase':'initial','problem_id':pid},cp['prompt_ids'],384,item['initial_seed'],checkpoint=True,row_call=cp['initial'])
        m.online_boundary(initial,256,tokenizer)
        m.require(cp['task']==task and cp['retained_ids']==initial['token_ids'],'Initial checkpoint data mismatch')
        eligible=initial['finish_reason']=='checkpoint'
        if eligible:
            m.checkpoint({**cp,'index':index},item,initial,tokenizer)
            m.require(cp['eligibility_reason'] is None,'Eligible state marked ineligible')
            record=local[pid];event=ledger.events[ledger.cursor]
            g=ledger.take({'phase':'local_judge','problem_id':pid,'temperature':0},event['prefix_ids'],192,20260927+index,row_call=record['generation'])
            # The recorded prompt must contain exactly the frozen rubric payload.
            rubric=m.read_json(run/'rubric.json')['questions']
            expected_prompt=judge_prompt(cp['state'],rubric)
            m.require(event['prefix_ids']==chat(expected_prompt),'Local judge prompt differs from exact state/rubric template')
            parsed=None
            try:
                if g['finish_reason']!='timeout':parsed=parse_probabilities(g['text'],rubric)
            except (ValueError,TypeError):pass
            m.require(record.get('probabilities')==parsed,'Local parsed features differ from generation')
            m.require(bool(record.get('error'))==(parsed is None),'Local failure status mismatch')
            expected_local={'usd':0.,'generated_tokens':g['generated_tokens'],'prompt_tokens':g['prompt_tokens'],'service_seconds':g['elapsed_seconds']}
            expected_jev={'usd':cp['jev'].get('input_cost_usd'),'service_seconds':cp['jev'].get('elapsed_seconds'),'generated_tokens':0}
            m.require(decisions[pid]['acquisition_costs']['cheap_tfidf_plus_local']==expected_local,'Local acquisition accounting mismatch')
            m.require(decisions[pid]['acquisition_costs']['cheap_tfidf_plus_jev']==expected_jev,'Jev acquisition accounting mismatch')
            document,numeric=screen.observable_features(cp,1024)
            expected_actions={'always_continue':'continue'}
            for name,controller in controllers.items():
                values=numeric
                if name.endswith('plus_jev'):values=np.r_[numeric,screen.semantic_features(cp.get('jev'))]
                elif name.endswith('plus_local'):values=np.r_[numeric,screen.semantic_features(record)]
                p=controller.predict([document],[values])[0]
                np.testing.assert_allclose(p,predictions[pid]['probabilities'][name],atol=1e-12,rtol=1e-12)
                expected_actions[name]=m.ACTIONS[int(p.argmax())]
            m.require(decisions[pid]['decisions']==expected_actions,'Frozen prediction/decision disagreement')
            predicted_count+=1
        else:
            m.require(all(a=='continue' for a in decisions[pid]['decisions'].values()),'Ineligible state intervened on')
            m.require(predictions[pid]['probabilities']=={} and pid not in local,'Ineligible state received judge/prediction')
        free={'usd':0.,'generated_tokens':0,'service_seconds':0.}
        for policy in ('always_continue','cheap_tfidf'):
            m.require(decisions[pid]['acquisition_costs'][policy]==free,'Cheap policy charged semantic acquisition')
        if not eligible:
            m.require(all(v==free for v in decisions[pid]['acquisition_costs'].values()),'Ineligible episode acquisition cost mismatch')
        selected=[r for r in rows if r['problem_id']==pid]
        for row in sorted(selected,key=lambda r:m.ACTIONS.index(r['action'])):
            audit_selected_rollout(row,item,cp,config,tokenizer,constants,ledger)
            if tasks['verify'](task,row['text']) != row['outcome']:
                outcome_disagreements.append({'problem_id':pid,'action':row['action']})
    m.require(ledger.cursor==len(ledger.events),'Unconsumed generation work')
    m.require(not outcome_disagreements,'Independent outcome disagreement')
    files=[p for p in run.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')]
    return {'status':'passed_prospective_audit','ready_for_analysis':True,'assignment':assignment,
            'predicted_checkpoints':predicted_count,'durable_calls':len(ledger.events),'outcome_disagreements':0,
            'input_sha256':{p.name:m.digest(p.read_bytes()) for p in files},
            'audit_source_sha256':m.digest(Path(__file__).read_bytes()),
            'limitations':['No model inference replay; call/token/state contracts are reconstructed.',
                          'Hosted feature content is measured input, not certified truth.',
                          'Decision ordering uses recorded timestamps and durable record hashes; not external timestamp attestation.',
                          'Weight files are not rehashed; tokenizer metadata is checked.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Audit output exists')
    report=audit(a.run,a.tokenizer)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','assignment','durable_calls','outcome_disagreements')}))
