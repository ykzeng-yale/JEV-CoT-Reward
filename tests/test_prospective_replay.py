import importlib.util
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
try:
    from prospective_replay import audit_selected_rollout, ProspectiveLedger
finally:
    sys.path.pop(0)


def test_completed_prefix_replay_rejects_hidden_continuation_or_text_change():
    cp={'initial':{'finish_reason':'stop','text':'FINAL: 1'}}
    row={'action':'continue','calls':[],'overhead':[],'text':'FINAL: 1',
         'generated_tokens':0,'prompt_tokens_processed':0,'elapsed_seconds':0}
    assert audit_selected_rollout(row,{},cp,{},None,{},None)==0
    with pytest.raises(ValueError):
        audit_selected_rollout({**row,'text':'FINAL: 2'},{},cp,{},None,{},None)
    with pytest.raises(ValueError):
        audit_selected_rollout({**row,'generated_tokens':1},{},cp,{},None,{},None)


def test_context_adapter_retains_action_problem_and_seed():
    class Ledger:
        def take(self,context,prefix,cap,seed,**kwargs):
            assert context=={'phase':'continuation','problem_id':'p','action':'repair'}
            assert (prefix,cap,seed)==([1],2,3)
            return 'checked'
    assert ProspectiveLedger(Ledger()).take({'phase':'continuation','problem_id':'p','action':'repair','index':0,'repeat':0},[1],2,3)=='checked'
