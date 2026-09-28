from jev_control.guard_adaptation import run
from jev_control.mlx_backend import Generation


def test_trigger_pool_budget_temperatures_and_decision_before_resume():
    calls=[];records=[]
    def generate(prefix,cap,seed,temp):
        # Increasing boundary entropy first permits a trigger at observation 11.
        n=len(calls);entropy=float(n+1)
        calls.append((list(prefix),cap,seed,temp))
        if n>=14:assert any(kind=='decisions' and row['trigger'] for kind,row in records)
        return Generation([65]*cap,'x'*cap,len(prefix),cap,-1.,entropy,.1,'length',0.,'fixture',seed,
                          [-1.]*cap,[entropy]*cap)
    result=run([1],[2]*256,256,generate,lambda s:[3]*len(s),lambda k,r:records.append((k,r)))
    triggered=[r for r in result['decisions'] if r['trigger']]
    assert triggered and triggered[0]['boundary']==11
    assert [calls[i][3] for i in (11,12,13)]==[0.,.6,1.5]
    assert triggered[0]['spent_after_pool']-triggered[0]['spent_before_pool']==300
    assert result['generated_tokens']==2048
    assert sum(r['generation']['generated_tokens'] for r in result['calls'])+256==2048
    # Injected instructions are prompt/history, not falsely called model output.
    assert result['calls'][-1]['context']=='final'


def test_terminal_segment_no_forced_resume_or_branch():
    def generate(prefix,cap,seed,temp):
        return Generation([65],'FINAL: 1',len(prefix),1,-1.,1.,.1,'stop',0.,'fixture',seed,[-1.],[1.])
    result=run([1],[],0,generate,lambda s:[],lambda k,r:None)
    assert len(result['calls'])==1 and result['generated_tokens']==1 and result['decisions']==[]


def test_answer_marker_does_not_truncate_unfinished_segment():
    count=0
    def generate(prefix,cap,seed,temp):
        nonlocal count
        count+=1
        return Generation([65],'FINAL:' if count==1 else '42',len(prefix),1,-1.,1.,.1,
                          'length' if count==1 else 'stop',0.,'fixture',seed,[-1.],[1.])
    result=run([1],[],0,generate,lambda s:[],lambda k,r:None,branching=False)
    assert count==2 and result['generated_tokens']==2
