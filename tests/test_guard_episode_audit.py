import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
from jev_control.guard_adaptation import run
from jev_control.mlx_backend import Generation
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_guard_episode import audit_episode


def fixture(branching=True):
    count=0
    def generate(prefix,cap,seed,temp):
        nonlocal count
        count+=1
        h=hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()
        return Generation([65]*cap,'a'*cap,len(prefix),cap,-1.,float(count),.1,'length',0.,h,seed,[-1.]*cap,[float(count)]*cap)
    encode=lambda s:[3]*len(s)
    return run([1],[2]*256,256,generate,encode,lambda k,r:None,branching=branching),encode

@pytest.mark.parametrize('branching',[True,False])
def test_independent_budget_and_prefix_reconstruction(branching):
    result,encode=fixture(branching)
    assert audit_episode(result,[1],[2]*256,256,encode,branching=branching)['generated_tokens']==2048

@pytest.mark.parametrize('field,value',[('requested_tokens',999),('temperature',2.),('prefix_ids',[1]),('spent_generated_tokens',0),('seed',999)])
def test_call_tampering_fails(field,value):
    result,encode=fixture();result['calls'][11][field]=value
    with pytest.raises(ValueError):audit_episode(result,[1],[2]*256,256,encode)

def test_winner_tampering_fails():
    result,encode=fixture();next(d for d in result['decisions'] if d['trigger'])['selected']=2
    with pytest.raises(ValueError):audit_episode(result,[1],[2]*256,256,encode)
