import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_branch_qualification as runner
from jev_control.mlx_backend import Generation


@pytest.mark.parametrize('eligible',[True,False])
def test_runner_preserves_all_enrollment_and_records_checkpoint_before_pool(tmp_path,monkeypatch,eligible):
    monkeypatch.setattr(runner,'model_identity',lambda p:{'fixture':True})
    class Backend:
        quantization_config={'bits':4}
        def __init__(self,*a,**kw):pass
        def encode_chat(self,messages):return [1]
        def generate(self,prefix,cap,seed,**kw):
            return Generation([65],'A',1,1,0.,0.,.1,'checkpoint' if eligible else 'stop',0.,'fixture',seed)
    monkeypatch.setattr(runner,'MLXBackend',Backend)
    def checkpoint(backend,task,prompt,g,*args):
        if not eligible:return None,'completed_before_checkpoint'
        return {'problem_id':task['id'],'initial':g.to_dict(),'retained_ids':g.token_ids,'prompt_ids':prompt},None
    monkeypatch.setattr(runner,'checkpoint_from_initial',checkpoint)
    output=tmp_path/'out'
    def pool(backend,task,cp,config,index,record,rollout,**kwargs):
        assert task['id'] in (output/'checkpoints.jsonl').read_text()
        record('decisions',{'problem_id':task['id'],'choices':{}})
        rows=[{'problem_id':task['id'],'fixture':i} for i in range(8)]
        for row in rows:record('outcomes',row)
        return {},rows
    monkeypatch.setattr(runner,'collect_pool',pool)
    runner.run(SimpleNamespace(output=output,model='fixture',wall_seconds=60))
    summary=json.loads((output/'summary.json').read_text())
    assert summary['status']=='complete' and summary['completed_problems']==8
    assert summary['outcomes']==(64 if eligible else 0)
    assert summary['eligible_problems']==(8 if eligible else 0)
    assert len((output/'skipped.jsonl').read_text().splitlines())==(0 if eligible else 8)
