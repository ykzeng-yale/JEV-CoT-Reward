import copy,importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('factorial',Path(__file__).parents[1]/'scripts/analyze_recoma_postphase_factorial_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture():
    tasks=[{'unit_id':f'{s}_{d}','scenario':f'family{s}'} for s in range(8) for d in range(3)]
    ids=['original_30','original_90','conservative_30','conservative_90'];design={'tasks':tasks,'arms':[{'arm_id':a} for a in ids],'_sha256':'bound'}
    arms={a:{'audit_status':'PASS','design_arm_id':a,'design_sha256':'bound','per_task':[{'unit_id':t['unit_id'],'official_success':False,'scientific_failure':False,'prompt_tokens':10,'generated_tokens':2,'model_calls':1,'environment_actions':1,'model_service_seconds':0.2} for t in tasks]} for a in ids};return design,arms
def test_zero_success_is_complete_negative_not_failure():
    d,a=fixture();r=m.analyze(d,a);assert r['primary']['ties']==24 and not r['practical_primary_margin_met']
def test_paired_opposing_effects_and_failure_adjustment():
    d,a=fixture();a['conservative_90']['per_task'][0]['official_success']=True;a['original_90']['per_task'][1]['official_success']=True;a['conservative_90']['per_task'][2].update(official_success=True,scientific_failure=True)
    r=m.analyze(d,a);assert (r['primary']['wins'],r['primary']['losses'],r['primary']['ties'])==(1,1,22)
@pytest.mark.parametrize('bad',['missing','duplicate','binding','boolean','negative','nan'])
def test_rejects_invalid_or_survivor_evidence(bad):
    d,a=fixture();v=a['original_30'];r=v['per_task'][0]
    if bad=='missing':v['per_task'].pop()
    elif bad=='duplicate':v['per_task'].append(copy.deepcopy(r))
    elif bad=='binding':v['design_sha256']='other'
    elif bad=='boolean':r['official_success']=1
    elif bad=='negative':r['prompt_tokens']=-1
    else:r['model_service_seconds']=float('nan')
    with pytest.raises(ValueError):m.analyze(d,a)
