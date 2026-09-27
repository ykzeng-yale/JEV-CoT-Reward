from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from jev_control.branch_experiment import collect_pool
from jev_control.mlx_backend import Generation


@pytest.mark.parametrize('use_jev',[False,True])
def test_pool_is_charged_in_full_and_decisions_precede_shadow_outcomes(use_jev):
    class Backend:
        context={}
        def decode(self,ids):return ''.join(map(chr,ids))
        def generate(self,prefix,cap,seed,**kwargs):
            j=self.context['candidate'];ids=[65+j]*4
            return Generation(ids,self.decode(ids),len(prefix),4,-float(j+1),float(3-j),.1,
                              'stop' if j==2 else 'length',0.,'fixture',seed)
    backend=Backend();events=[];calls=[]
    config={'budget':100,'final_reserve':10,'task_seed':7,'candidate_count':3,'candidate_tokens':4,'judge_tokens':8,'repeats':2}
    cp={'prompt_ids':[1],'retained_ids':[2]*10,'initial':{'generated_tokens':10,'prompt_tokens':1,'elapsed_seconds':.1}}
    def record(kind,row):events.append((kind,row))
    def judge(*args):return {'choice':None,'error':'bad','generation':{'generated_tokens':5,'prompt_tokens':20,'elapsed_seconds':.2}}
    def rollout(backend,task,prompt,retained,action,seed,remaining,reserve):
        assert any(k=='decisions' for k,r in events)
        calls.append(remaining)
        return {'action':'continue','seed':seed,'text':'fixture','outcome':{'success':False},
                'generated_tokens':remaining,'prompt_tokens_processed':len(prompt+retained),'elapsed_seconds':.1,'calls':[],'overhead':[]}
    from jev_control.jev import JevClient, MODEL
    client=SimpleNamespace(evaluate=lambda state,questions: {'response':{'answers':{'candidate':{'choice':'1'}},'usage':{'input_tokens':100}},'input_cost_usd':.0000042})
    config['jev']=use_jev
    decisions,rows=collect_pool(backend,{'id':'p','prompt':'task'},cp,config,0,record,rollout,judge,jev_client=client)
    assert len(rows)==8 and calls==[90,78,78,90,78,78]
    assert decisions['choices']['local_semantic']==decisions['choices']['likelihood']==0
    assert decisions['choices']['entropy_reduction']==2
    assert all(r['episode_generated_tokens']==100 for r in rows if r['candidate']!=2)
    assert all(r['episode_generated_tokens']==22 and r['calls']==[] for r in rows if r['candidate']==2)
    first_outcome=next(i for i,(k,r) in enumerate(events) if k=='outcomes')
    assert next(i for i,(k,r) in enumerate(events) if k=='decisions')<first_outcome
    assert len([r for k,r in events if k=='candidates'])==3

    if use_jev:
        assert decisions["choices"]["jev_semantic"]==1
        assert next(i for i,(k,r) in enumerate(events) if k=="jev_judge")<first_outcome
        assert all(r["jev_accounted_usd"]==.0000042 for r in rows if r["candidate"] is not None)
