import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
from jev_control.prospective import POLICIES

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
try:
    from analyze_prospective import analyze
finally:sys.path.pop(0)


def test_paired_analysis_counts_shared_results_once_per_policy_and_rejects_stale_audit(tmp_path):
    names=['manifest.json','summary.json','schedule.json','checkpoints.jsonl','decisions.jsonl','outcomes.jsonl',
           'predictions.jsonl','local_judge.jsonl','generation_started.jsonl','generation_events.jsonl','rubric.json']
    for name in names:(tmp_path/name).write_text('{}\n')
    schedule=[{'task':{'id':f'p{i}','family':'a' if i%2 else 'b'}} for i in range(24)]
    (tmp_path/'schedule.json').write_text(json.dumps(schedule))
    cost={'generated_tokens':10,'prompt_tokens_processed':20,'model_service_seconds':1.,
          'acquisition':{'usd':0.,'service_seconds':0.,'generated_tokens':0}}
    rows=[{'problem_id':s['task']['id'],'policies':list(POLICIES),'action':'continue',
           'outcome':{'success':i%2==0},'hypothetical_deployment_costs':{p:cost for p in POLICIES}}
          for i,s in enumerate(schedule)]
    (tmp_path/'outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'generation_events.jsonl').write_text(json.dumps({'generation':{'generated_tokens':240,'elapsed_seconds':24}})+'\n')
    audit={'status':'passed_prospective_audit','ready_for_analysis':True,
           'input_sha256':{n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in names}}
    audit_path=tmp_path/'audit.json';audit_path.write_text(json.dumps(audit))
    result=analyze(tmp_path,audit_path,draws=20)
    assert result['policy_episodes']==96 and result['actual_shared_action_continuations']==24
    assert result['collection_generated_tokens']==240
    assert all(p['successes']==12 for p in result['policies'].values())
    assert all(c['mean']==0 and c['discordant_problems']==0 for c in result['paired_contrasts'].values())
    (tmp_path/'outcomes.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='changed after audit'):analyze(tmp_path,audit_path)
