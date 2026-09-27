import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from analyze_action_qualification import analyze


def test_analysis_clusters_repeats_charges_prefix_and_requires_unchanged_audit(tmp_path):
    actions=['continue','sham','suffix_repair','recheck','segment_repair']
    cps=[{'problem_id':str(i),'task':{'family':'a' if i%2 else 'b'},
          'initial':{'generated_tokens':10,'prompt_tokens':20,'elapsed_seconds':.1}} for i in range(8)]
    rows=[{'problem_id':str(i),'action':a,'repeat':r,'outcome':{'success':bool(i%2),'reason':'fixture'},
           'generated_tokens':30,'prompt_tokens_processed':40,'elapsed_seconds':.2,
           'preparation':{'removed_tokens':2,'inserted_tokens':3},'calls':[{'finish_reason':'stop'}]}
          for i in range(8) for a in actions for r in range(2)]
    files={'manifest.json':json.dumps({'config':{'actions':actions,'repeats':2,'problems':8}}),
           'checkpoints.jsonl':'\n'.join(map(json.dumps,cps)), 'outcomes.jsonl':'\n'.join(map(json.dumps,rows)),
           'schedule.json':'[]','generation_events.jsonl':''}
    for name,data in files.items():(tmp_path/name).write_text(data)
    report={'status':'passed_action_qualification_audit','ready_for_analysis':True,
            'input_sha256':{n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in files}}
    path=tmp_path/'audit.json';path.write_text(json.dumps(report))
    result=analyze(tmp_path,path)
    assert result['eligible_problems']==8 and result['episodes']==80
    for a in result['actions'].values():
        assert a['episodes']==16 and a['successes']==8
        assert a['problem_weighted_success']['n_problems']==8
        assert a['mean_episode_generated_tokens']==40
        assert a['mean_episode_processed_prompt_tokens']==60
    (tmp_path/'outcomes.jsonl').write_text('tamper')
    with pytest.raises(ValueError,match='changed'):analyze(tmp_path,path)
