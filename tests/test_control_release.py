import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from release_control_outcomes import export, sha
from check_control_release import check
from jev_control.tasks import make_task, verify


def test_release_rechecks_labels_and_rejects_tampering(tmp_path):
    run=tmp_path/'run';run.mkdir()
    task=make_task(0);text='FINAL: A->H'
    (run/'schedule.json').write_text(json.dumps([{'task':task}]))
    (run/'outcomes.jsonl').write_text(json.dumps({'problem_id':task['id'],'action':'continue',
        'text':text,'outcome':verify(task,text),'private_field':'must be omitted'})+'\n')
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps({'ready_for_analysis':True,'status':'passed_prospective_audit',
        'input_sha256':{n:sha(run/n) for n in ('schedule.json','outcomes.jsonl')}}))
    dest=tmp_path/'public';export(run,audit,dest)
    assert check(dest)['outcome_disagreements']==0
    assert 'private_field' not in (dest/'outcomes.jsonl').read_text()
    (dest/'outcomes.jsonl').write_text('{}')
    with pytest.raises(ValueError,match='Release changed'):check(dest)
    (run/'outcomes.jsonl').write_text('{}')
    with pytest.raises(ValueError,match='Stale audit'):export(run,audit,tmp_path/'other')
