import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_action_qualification as runner
from jev_control.mlx_backend import Generation


@pytest.mark.parametrize('eligible',[True,False])
def test_complete_grid_and_no_replacement_for_ineligible(tmp_path,monkeypatch,eligible):
    monkeypatch.setattr(runner,'model_identity',lambda p:{'fixture':True})
    class Backend:
        quantization_config={'bits':4}
        def __init__(self,*a,**kw):pass
        def encode_chat(self,messages):return [1]
        def encode_text(self,text):return list(map(ord,text))
        def decode(self,ids):return ''.join(map(chr,ids))
        def generate(self,prefix,max_tokens,seed,**kw):
            ids=list(map(ord,'earlier\n\nnext\n'))
            return Generation(ids,self.decode(ids),len(prefix),len(ids),0.,0.,.01,
                'checkpoint' if eligible else 'stop',0.,'fixture',seed)
    monkeypatch.setattr(runner,'MLXBackend',Backend)
    def checkpoint(backend,task,prompt,initial,*args):
        if not eligible:return None,'completed_before_checkpoint'
        return {'problem_id':task['id'],'retained_ids':initial.token_ids,'prompt_ids':prompt,
                'initial':initial.to_dict(),'sha256':'fixture'},None
    monkeypatch.setattr(runner,'checkpoint_from_initial',checkpoint)
    seen=[]
    def rollout(backend,task,prompt,prepared,action,seed,remaining,reserve):
        assert action=='continue' and remaining==2048-14 and reserve==128
        seen.append((task['id'],prepared))
        return {'generated_tokens':5,'outcome':{'success':False},'action':action}
    monkeypatch.setattr(runner,'rollout',rollout)
    output=tmp_path/'out'
    runner.run(SimpleNamespace(output=output,model='fixture',wall_seconds=60))
    summary=json.loads((output/'summary.json').read_text())
    assert summary['status']=='complete'
    assert len(seen)==(80 if eligible else 0)
    assert summary['skipped_problems']==(0 if eligible else 8)
    if eligible:
        rows=[json.loads(s) for s in (output/'outcomes.jsonl').read_text().splitlines()]
        assert all(set(r['action'] for r in rows if r['problem_id']==pid)==set(runner.ACTIONS) for pid,_ in seen)
