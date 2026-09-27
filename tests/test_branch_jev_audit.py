import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_branch_jev import audit_record,MODEL


def fixture():
    req={'state':'observable task only','questions':{'candidate':{'type':'choice','instructions':'choose','criteria':{'0':'a','1':'b','2':'c'}}}}
    digest=hashlib.sha256(json.dumps({'model':MODEL,**req},sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    response={'model':MODEL,'answers':{'candidate':{'type':'choice','choice':'1','probabilities':{'0':.2,'1':.7,'2':.1},'confidence':.5}},'usage':{'input_tokens':100}}
    row={'request':req,'recorded_unix':2.,'acquisition_seconds':.2,'error':None,'choice':1,'input_tokens':100,'accounted_usd':.0000042,
         'result':{'response':response,'input_cost_usd':.0000042,'request_sha256':digest,'cache_hit':False,'elapsed_seconds':.2}}
    return row,copy.deepcopy(req)


def test_success_and_failed_fallback():
    row,req=fixture();assert audit_record(row,req,1.,3.)==1
    row.update(error='TimeoutError',choice=None,input_tokens=None,accounted_usd=.01);row.pop('result')
    assert audit_record(row,req,1.,3.) is None

@pytest.mark.parametrize('path,value',[
    (('request','state'),'gold answer injected'),(('recorded_unix',),4.),(('recorded_unix',),float('nan')),
    (('acquisition_seconds',),-1), (('choice',),True), (('input_tokens',),True),
    (('accounted_usd',),0.), (('result','input_cost_usd'),0.),(('result','request_sha256'),'bad'),
    (('result','cache_hit'),1), (('result','response','model'),'latest'),
    (('result','response','answers','candidate','choice'),'0'),
    (('result','response','answers','candidate','confidence'),float('inf')),
    (('result','response','answers','candidate','probabilities','0'),.3),
    (('result','response','usage','input_tokens'),-1)])
def test_record_tampering_rejected(path,value):
    row,req=fixture();target=row
    for k in path[:-1]:target=target[k]
    target[path[-1]]=value
    with pytest.raises(ValueError):audit_record(row,req,1.,3.)
