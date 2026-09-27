import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_branch_qualification import BranchLedger


def test_branch_audit_requires_decision_before_call_and_preserves_exact_context():
    class Ledger:
        events=[{'started_unix':12.}];cursor=0
        def take(self,context,*args,**kwargs):return context,args,kwargs
    context={'phase':'shadow_continuation','candidate':2,'problem_id':'p','repeat':1}
    bridge=BranchLedger(Ledger(),context,11.)
    got=bridge.take({'index':9,'action':'continue'},[1,2],30,101,row_call={'x':1})
    assert got==(context,([1,2],30,101),{'row_call':{'x':1}})
    with pytest.raises(ValueError,match='before selection'):
        BranchLedger(Ledger(),context,13.).take({},[],1,1)
