import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from analyze_branch_qualification import analyze


@pytest.mark.parametrize('jev',[False,True])
def test_shared_policy_accounting_charges_local_only_and_clusters_repeats(tmp_path,jev):
    selectors=['uniform','likelihood','entropy_reduction','local_semantic']
    if jev: selectors.append('jev_semantic')
    repeats=4 if jev else 2
    cps=[{'problem_id':str(i),'task':{'family':str(i%2)}} for i in range(8)]
    rows=[]
    for i in range(8):
        for repeat in range(repeats):
            for j in (None,0,1,2):
                rows.append({'problem_id':str(i),'repeat':repeat,'candidate':j,
                    'selectors':['continue'] if j is None else selectors if j==0 else [],
                    'outcome':{'success':i%2==0},'episode_generated_tokens':100,'episode_prompt_tokens':200,
                    'episode_service_seconds':2.,'selector_generated_tokens':10,'selector_prompt_tokens':20,'selector_service_seconds':.5,'jev_accounted_usd':.001,'jev_input_tokens':100,'jev_acquisition_seconds':.2})
    records={'outcomes':rows,'checkpoints':cps,
        'decisions':[{'problem_id':str(i),'choices':{p:0 for p in selectors},'failures':{}} for i in range(8)],
        'candidates':[{'problem_id':str(i),'generation':{'text':str(j)}} for i in range(8) for j in range(3)],
        'generation_events':[{'status':'complete','generation':{'generated_tokens':100,'elapsed_seconds':1.}}]}
    for name,items in records.items():(tmp_path/(name+'.jsonl')).write_text('\n'.join(map(json.dumps,items)))
    (tmp_path/'manifest.json').write_text(json.dumps({'config':{'selectors':selectors,'repeats':repeats,'jev':jev,'candidate_count':3,'problems':8}}))
    (tmp_path/'schedule.json').write_text('[]')
    audit={'status':'passed_branch_qualification_audit','ready_for_analysis':True,'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}}
    path=tmp_path/'audit.json';path.write_text(json.dumps(audit));result=analyze(tmp_path,path)
    assert result['hypothetical_policy_episodes']==8*repeats*(len(selectors)+1) and result['unique_outcomes']==8*repeats*4
    assert result['policies']['local_semantic']['mean_selector_generated_tokens']==10
    assert result['policies']['likelihood']['mean_selector_generated_tokens']==0
    assert result['policies']['likelihood']['mean_total_processed_prompt_tokens']==200
    assert result['policies']['local_semantic']['mean_total_processed_prompt_tokens']==220
    assert result['policies']['local_semantic']['problem_weighted_success']['n_problems']==8

    if jev:
        assert result["policies"]["jev_semantic"]["mean_selector_generated_tokens"]==0
        assert result["policies"]["jev_semantic"]["mean_jev_accounted_usd"]==.001
        assert result["policies"]["local_semantic"]["mean_jev_accounted_usd"]==0
        assert result["split_repeat_diagnostic"][0]["selection_repeats"]==[0,1]
        assert result["split_repeat_diagnostic"][0]["evaluation_repeats"]==[2,3]
        assert "jev_semantic_minus_likelihood" in result["paired_contrasts"]
