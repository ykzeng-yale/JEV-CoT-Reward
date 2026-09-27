"""Exercise runner control flow without models, network, or new study tasks."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from jev_control.mlx_backend import Generation
from jev_control.prospective import POLICIES

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    spec=importlib.util.spec_from_file_location('prospective_runner',SCRIPTS/'run_prospective.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
finally:
    sys.path.remove(str(SCRIPTS))


def setup(tmp_path,monkeypatch,mode):
    output=tmp_path/'run';training=tmp_path/'training';training.mkdir()
    (training/'manifest.json').write_text(json.dumps({'model_identity':{'weight_revision_sha256':'weights'}}))
    frozen=tmp_path/'frozen';frozen.mkdir();(frozen/'manifest.json').write_text('{}')
    manifest={'artifact_sha256':'artifact','training_problem_ids':['old'],
              'training_inputs_sha256':{'manifest.json':runner.file_sha256(training/'manifest.json')}}
    monkeypatch.setattr(runner,'load_frozen',lambda _: ({},manifest))
    monkeypatch.setattr(runner,'model_identity',lambda _: {'weight_revision_sha256':'weights'})
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'protocol':'synthetic','problems':2,'task_seed':991}))
    # Source snapshot requires a repository-relative config; use real config and
    # leave the 24-row enrollment intact, with lightweight generated fixtures.
    monkeypatch.setattr(runner,'make_task',lambda i,s:{'id':f'new-{i}','family':'fixture','prompt':'fixture','data':{}})
    class Backend:
        quantization_config={'bits':4}
        def __init__(self,*a,**kw):pass
        def encode_chat(self,messages):return [1]
        def generate(self,prefix,max_tokens,seed,**kw):
            return Generation([2],'fixture',len(prefix),1,-1.,1.,.1,'stop' if mode=='early' else 'checkpoint',0.,'hash',seed)
    monkeypatch.setattr(runner,'MLXBackend',Backend)
    def checkpoint(backend,task,prompt,initial,target,cap):
        if mode=='early':return None,'completed_before_checkpoint'
        return {'problem_id':task['id'],'task':task,'prompt_ids':prompt,'retained_ids':initial.token_ids,
                'initial':initial.to_dict(),'state':{'task':'fixture','history':'','latest_segment':'fixture'},'sha256':'hash'},None
    monkeypatch.setattr(runner,'checkpoint_from_initial',checkpoint)
    import jev_control.jev as jev
    class Client:
        def __init__(self,**kw):pass
        def evaluate(self,*a):
            if mode=='failure':raise RuntimeError('synthetic acquisition failure')
            return {'input_cost_usd':.001,'elapsed_seconds':.2}
    monkeypatch.setattr(jev,'JevClient',Client)
    monkeypatch.setattr(runner,'local_features',lambda *a:{'generation':{'generated_tokens':2,'prompt_tokens':5,'elapsed_seconds':.1}})
    monkeypatch.setattr(runner,'predict_actions',lambda *a: (dict(zip(POLICIES,['continue','repair','repair','continue'])),{}))
    monkeypatch.setattr(runner,'verify',lambda *a:{'success':True})
    calls=[]
    def rollout(backend,task,prompt,retained,action,seed,remaining,reserve):
        decisions=[json.loads(line) for line in (output/'decisions.jsonl').read_text().splitlines()]
        assert decisions[-1]['problem_id']==task['id']
        assert set(decisions[-1]['decisions'])==set(POLICIES)
        calls.append((task['id'],action))
        return {'action':action,'seed':seed,'generated_tokens':3,'prompt_tokens_processed':5,
                'elapsed_seconds':.2,'outcome':{'success':False},'calls':[],'text':'fixture','overhead':[]}
    monkeypatch.setattr(runner,'rollout',rollout)
    args=SimpleNamespace(model='fixture',controllers=frozen,training_run=training,output=output,wall_seconds=60)
    return args,calls


@pytest.mark.parametrize('mode',['early','eligible'])
def test_full_runner_enrollment_ordering_and_common_result(tmp_path,monkeypatch,mode):
    args,calls=setup(tmp_path,monkeypatch,mode)
    runner.run(args)
    summary=json.loads((args.output/'summary.json').read_text())
    assert summary['status']=='complete' and summary['completed_problems']==24
    rows=[json.loads(s) for s in (args.output/'outcomes.jsonl').read_text().splitlines()]
    assert sum(len(r['policies']) for r in rows)==96
    assert len(calls)==(0 if mode=='early' else 48)
    assert len(rows)==(24 if mode=='early' else 48)
    if mode=='early':assert all(r['generated_tokens']==0 for r in rows)


def test_acquisition_failure_preserves_checkpoint_and_no_outcomes(tmp_path,monkeypatch):
    args,calls=setup(tmp_path,monkeypatch,'failure')
    with pytest.raises(RuntimeError,match='synthetic acquisition'):
        runner.run(args)
    assert not calls
    assert (args.output/'checkpoint_attempts.jsonl').read_text()
    assert not (args.output/'outcomes.jsonl').read_text()
    summary=json.loads((args.output/'summary.json').read_text())
    assert summary['status']=='failed' and summary['completed_problems']==0
