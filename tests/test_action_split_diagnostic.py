import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from analyze_action_qualification import split_repeat_diagnostic


def test_selection_cannot_see_evaluation_winner():
    table={('p',a,r):{'outcome':{'success':v}} for a,vs in [('continue',[1,1,0,0]),('repair',[0,0,1,1])] for r,v in enumerate(vs)}
    result=split_repeat_diagnostic(table,['p'],['continue','repair'],4)
    assert result[0]['selected_action_counts']=={'continue':1}
    assert result[0]['selected_success']==0
    assert result[1]['selected_action_counts']=={'repair':1}
    assert result[1]['paired_difference']==-1


def test_ties_use_frozen_order_not_test_success():
    table={('p',a,r):{'outcome':{'success':v}} for a,vs in [('continue',[0,0,0,0]),('repair',[0,0,1,1])] for r,v in enumerate(vs)}
    result=split_repeat_diagnostic(table,['p'],['continue','repair'],4)
    assert result[0]['selected_action_counts']=={'continue':1}
    assert split_repeat_diagnostic(table,['p'],['continue','repair'],2)==[]
