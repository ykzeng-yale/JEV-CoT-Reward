import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from audit_guard_qualification import bind_records


def fixture():
    calls=[{'context':c} for c in ['segment','candidate_0','candidate_1','candidate_2','segment']]
    decisions=[{'trigger':True}]
    row={'calls':calls,'decisions':decisions,'recorded_unix':20.}
    records=[{**c,'problem_id':'p','policy':'g','recorded_unix':2.*i+1} for i,c in enumerate(calls)]
    ds=[{'trigger':True,'recorded_unix':7.5,'problem_id':'p','policy':'g'}]
    return row,records,ds,[0.,2.,4.,6.,8.]

def test_binding_and_order():bind_records(*fixture())

@pytest.mark.parametrize('kind',['late_decision','early_decision','late_call','wrong_copy'])
def test_tampering_rejected(kind):
    row,records,ds,starts=fixture()
    if kind=='late_decision':ds[0]['recorded_unix']=8.5
    if kind=='early_decision':ds[0]['recorded_unix']=5.
    if kind=='late_call':records[0]['recorded_unix']=3.
    if kind=='wrong_copy':records[1]['context']='wrong'
    with pytest.raises(ValueError):bind_records(row,records,ds,starts)
