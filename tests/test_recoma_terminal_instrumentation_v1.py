import ast,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
root=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location('patch',root/'scripts/build_recoma_terminal_instrumentation_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=root/'runs/recoma-final-six-hour-v3-control/production-inputs-complete-27983982/inputs/source'
pytestmark=pytest.mark.skipif(not source.is_dir(), reason='requires preserved private frozen runtime; no replacement source downloaded')
def method(text):
 tree=ast.parse(text);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='terminate_with_output');fn.returns=None
 for a in fn.args.args:a.annotation=None
 ns={'Optional':object,'Action':SimpleNamespace,'task_completed':lambda:False};exec(compile(ast.Module(body=[fn],type_ignores=[]),'test','exec'),ns);return ns
def test_actual_controller_submit_branch_preserves_return_and_skips_world_query():
 text=(source/'discoveryworld/agents/recoma/react_controller.py').read_text();ns=method(m.patch_controller(text));ns['task_completed']=lambda:(_ for _ in ()).throw(AssertionError('extra world query'))
 self=SimpleNamespace(get_react_node=lambda s:None,get_history=lambda n:[SimpleNamespace(action_json={'action':'SUBMIT','arg1':'answer','thought':'reason'})]);state=SimpleNamespace(data={})
 assert ns['terminate_with_output'](self,state,SimpleNamespace(output='unused'))=='answerreason';assert state.data['recoma_terminal_trigger']=='submit'
def test_actual_completion_branch_return_and_single_query_preserved():
 text=(source/'discoveryworld/agents/recoma/react_controller.py').read_text();ns=method(m.patch_controller(text));count=[];ns['task_completed']=lambda:count.append(1) or True
 self=SimpleNamespace(get_react_node=lambda s:None,get_history=lambda n:[SimpleNamespace(action_json={'action':'USE'})]);state=SimpleNamespace(data={});assert ns['terminate_with_output'](self,state,SimpleNamespace(output='world'))=='world';assert len(count)==1 and state.data['recoma_terminal_trigger']=='official_completion'
def test_anchor_mutation_rejected():
 with pytest.raises(ValueError):m.patch_controller('unrecognized source')

def test_actual_action_cap_checks_counter_once_and_preserves_stop():
    text=m.patch_action_cap((source/'discoveryworld/agents/recoma/discoveryworld_env_models.py').read_text())
    tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MaximumEnvironmentCalls');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='should_stop')
    for a in fn.args.args:a.annotation=None
    count=[];ns={'num_interactions':lambda:count.append(1) or 30,'logger':SimpleNamespace(warning=lambda x:None)};exec(compile(ast.Module(body=[fn],type_ignores=[]),'test','exec'),ns)
    state=SimpleNamespace(data={});assert ns['should_stop'](SimpleNamespace(max_env_calls=30),state,0,[]) is True;assert len(count)==1 and state.data['recoma_terminal_trigger']=='action_cap'
    count.clear();ns['num_interactions']=lambda:count.append(1) or 29;state=SimpleNamespace(data={});assert ns['should_stop'](SimpleNamespace(max_env_calls=30),state,0,[]) is False;assert len(count)==1 and not state.data

def test_durable_prediction_keeps_endpoint_and_stop_branch():
    text=m.patch_durable((source/'recoma/recoma/utils/task_accounting.py').read_text());ns={'__file__':__file__};exec(compile(text,'prepared','exec'),ns)
    ex=SimpleNamespace(unique_id='task');state=SimpleNamespace(data={'recoma_terminal_trigger':'submit','final_scorecard':[{'completedSuccessfully':False}]});pred=SimpleNamespace(example=ex,prediction='0.1',final_state=state)
    row=ns['prediction_record'](pred);assert row['terminal_trigger']=='submit' and not row['failure_adjusted_completed_successfully'] and row['metadata']['final_scorecard'][0]['completedSuccessfully'] is False
def test_format_and_infrastructure_hooks_compile_without_retry():
    search=m.patch_search((source/'recoma/recoma/search/search.py').read_text());compile(search,'prepared','exec');assert 'current_state.data["recoma_terminal_trigger"] = "format_failure"' in search
    durable=m.patch_durable((source/'recoma/recoma/utils/task_accounting.py').read_text());assert '"outcome_unknown": True' in durable;compile(durable,'prepared','exec')
