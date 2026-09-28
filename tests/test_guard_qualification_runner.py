import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_guard_qualification as runner
from jev_control.mlx_backend import Generation

@pytest.mark.parametrize('eligible',[True,False])
def test_complete_runtime_schedule_and_ineligible_preservation(tmp_path,monkeypatch,eligible):
    monkeypatch.setattr(runner,'model_identity',lambda p:{'fixture':True})
    class Backend:
        quantization_config={'bits':4}
        def __init__(self,*a,**kw):pass
        def encode_chat(self,messages):return [1]
        def encode_text(self,text):return [3]*len(text)
        def decode(self,ids):return 'not a verified answer'
        def generate(self,prefix,cap,seed,**kw):
            return Generation([65],'A',len(prefix),1,-1.,1.,.1,'stop',0.,'fixture',seed,[-1.],[1.])
    monkeypatch.setattr(runner,'MLXBackend',Backend)
    def checkpoint(backend,task,prompt,g,*args):
        if not eligible:return None,'completed_before_checkpoint'
        return {'problem_id':task['id'],'initial':g.to_dict(),'retained_ids':g.token_ids,'prompt_ids':prompt},None
    monkeypatch.setattr(runner,'checkpoint_from_initial',checkpoint)
    monkeypatch.setattr(runner,'scoped_generate',lambda b,p,c,s,t:b.generate(p,c,s))
    out=tmp_path/'out';runner.run(SimpleNamespace(output=out,model='fixture',wall_seconds=60))
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='complete' and summary['completed_problems']==4
    assert summary['outcomes']==(12 if eligible else 0)
    rows=[json.loads(s) for s in (out/'outcomes.jsonl').read_text().splitlines()]
    assert all(r['episode_generated_tokens']<=2048 for r in rows)
    if eligible:assert {r['policy'] for r in rows}=={'continue','segmented_sham','guard_adaptation'}
    assert len((out/'skipped.jsonl').read_text().splitlines())==(0 if eligible else 4)
