"""Source-bound pre-outcome visible census: real record gate + adverse visibility."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_recoma_discoveryworld_v3_audit import (FakeTokenizer, MODEL, fixture, make_failure,
                                                read_rows, run, write_rows)
from scripts.analyze_recoma_visible_instrument_census_v3 import (
    INSTRUMENTS, SOURCE_FILES, VERSION, census, checked_source, visible_state)
from scripts.audit_recoma_discoveryworld_v3 import digest_text,sha256

ANALYZER=Path(__file__).resolve().parents[1]/'scripts/analyze_recoma_visible_instrument_census_v3.py'


def prompt(objects=None,*,dialog=False,history='',known=True):
    objects=objects if objects is not None else [{'uuid':1,'name':'thermometer'},{'uuid':2,'name':'sample'}]
    view={'ui':{'inventoryObjects':objects[:1],'accessibleEnvironmentObjects':objects[1:],
                'dialog_box':{'is_in_dialog':dialog}}}
    text=history+'\nCurrent Environment Observation:\n```json\n'+json.dumps(view)+'\n```\n'
    if not dialog:
        text+='Valid Actions:\n```json\n'+json.dumps({'USE':{'args':['arg1','arg2']}} if known else {'WAIT':{}})+'\n```\n'
    return [{'role':'user','content':text}]


def add_census_source(inputs):
    source=inputs['source']
    files={
        SOURCE_FILES[0]:'Current Environment Observation:\n```json\n{{ observation }}\n```\n\nValid Actions:\n```json\n{{ known_actions}}\n```\n',
        SOURCE_FILES[1]:'''import copy,json
class Prompt:
    def populate_template_dictionary(self):
        observation=self.env.getAgentObservation(agentIdx=0)
        observationNoVision=copy.deepcopy(observation)
        param_dict={}
        param_dict["observation"]=json.dumps(observationNoVision,indent=4,sort_keys=True)
        return param_dict
''',
        SOURCE_FILES[3]:'''class Instruments:
    def __init__(self,world):
        Object.__init__(self,world,"microscope","microscope")
        Object.__init__(self,world,"PH meter","PH meter")
        Object.__init__(self,world,"radiation meter","radiation meter")
        Object.__init__(self,world,"spectrometer","spectrometer")
        Object.__init__(self,world,"thermometer","thermometer")
        Object.__init__(self,world,"densitometer","densitometer")
        Object.__init__(self,world,"proteomics meter","proteomics meter")
''',
    }
    for relative,content in files.items():
        path=source/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    ui=source/SOURCE_FILES[2]
    ui.write_text(ui.read_text()+'''    def renderObjectSelectionBoxJSON(self,objsInv,objsEnv):
        invOut=[{"uuid":obj.uuid,"name":obj.name} for obj in objsInv]
        envOut=[{"uuid":obj.uuid,"name":obj.name} for obj in objsEnv]
        return {"inventoryObjects":invOut,"accessibleEnvironmentObjects":envOut}
''')
    inputs['manifest']['endpoint_source_sha256'][SOURCE_FILES[2]]=sha256(ui)
    inputs['manifest']['visible_census_contract']={'version':VERSION,'analyzer_sha256':sha256(ANALYZER),
        'source_sha256':{relative:sha256(source/relative) for relative in SOURCE_FILES},
        'selection_uses_visible_fields_only':True,'selection_uses_terminal_outcomes':False,
        'recognized_instrument_names':list(INSTRUMENTS),'checkpoint_rule':'first_classifiable_candidate_per_task'}
    inputs['manifest']['artifact_sha256']={'audit_recoma_discoveryworld_v3.py':sha256(
        ANALYZER.parent/'audit_recoma_discoveryworld_v3.py')}


def setup(tmp_path):
    inputs=fixture(tmp_path,'full_panel');manifest=inputs['manifest']
    manifest.update(audit_scope='balanced_census',seeds=[0],timing_based_fallback=True,
                    selection_without_outcome_rates=True)
    manifest['task_instances']=[row for row in manifest['task_instances'] if row['random_seed']==0]
    ids={f"{row['scenario_name']}_{row['difficulty']}_0" for row in manifest['task_instances']}
    manifest['workers'][0]['task_ids']=[task for task in manifest['workers'][0]['task_ids'] if task in ids]
    inputs['runtime']['expected_task_ids']=manifest['workers'][0]['task_ids']
    for kind in ('outcomes','calls','records','events'):
        write_rows(inputs['paths'][kind],[row for row in read_rows(inputs['paths'][kind]) if row['task_id'] in ids])
    add_census_source(inputs)
    calls=read_rows(inputs['paths']['calls'])
    for call in calls:
        call['messages']=prompt()
        call['prompt_sha256']=digest_text(json.dumps(call['messages'],ensure_ascii=False,separators=(',',':')))
    write_rows(inputs['paths']['calls'],calls)
    return inputs


def analyze(inputs):
    audit=run(inputs)
    manifest_sha=digest_text(json.dumps(inputs['manifest'],sort_keys=True))
    audit.update(manifest_sha256=manifest_sha,auditor_sha256=sha256(ANALYZER.parent/'audit_recoma_discoveryworld_v3.py'))
    result=census(inputs['manifest'],[inputs['paths']['calls']],audit,inputs['source'],manifest_sha256=manifest_sha)
    return result,audit,manifest_sha


def test_complete_audited_census_keeps_denominator_and_disclaims_causal_value(tmp_path):
    inputs=setup(tmp_path);result,_,_=analyze(inputs)
    assert result['assigned_task_denominator']==24
    assert result['independent_template_difficulty_strata']==24
    assert result['counts']['tasks_with_observed_candidate']==24
    assert result['observed_candidate_task_fraction']==1
    assert result['uses_terminal_scorecards_or_success'] is False
    assert result['new_counterfactual_interventions']==0
    assert 'not proven legal pairs' in result['claim_boundary']
    assert 'successful measurements' in result['claim_boundary']
    assert 'sample' not in json.dumps(result)
    assert result['per_task'][0]['first_classifiable_candidate']['task_call_index']==0


def test_unsupported_or_historical_object_mentions_do_not_create_candidates():
    messages=prompt([{'uuid':9,'name':'chair'}],history='History: USE thermometer on sample; success claimed.')
    assert visible_state(messages)['candidate'] is False
    assert visible_state(prompt([{'uuid':1,'name':'thermometer'},{'uuid':1,'name':'thermometer'}]))['candidate'] is False
    assert visible_state(prompt(dialog=True))['candidate'] is False
    assert visible_state(prompt(known=False))['candidate'] is False
    assert visible_state(prompt([{'uuid':1,'name':'THERMOMETER'},{'uuid':2,'name':'sample'}]))['candidate'] is True


@pytest.mark.parametrize('malformation', ['uuid_bool','uuid_conflict','missing_dialog','missing_objects','extra_anchor','bad_json'])
def test_uncertain_visible_state_is_unknown_not_false_evidence(malformation):
    messages=prompt()
    if malformation=='uuid_bool': messages=prompt([{'uuid':True,'name':'thermometer'},{'uuid':2,'name':'sample'}])
    elif malformation=='uuid_conflict': messages=prompt([{'uuid':1,'name':'thermometer'},{'uuid':1,'name':'chair'}])
    elif malformation=='missing_dialog': messages[0]['content']=messages[0]['content'].replace('"is_in_dialog": false','"wrong_key": false')
    elif malformation=='missing_objects': messages[0]['content']=messages[0]['content'].replace('inventoryObjects','unknownObjects')
    elif malformation=='extra_anchor': messages[0]['content']+='\nCurrent Environment Observation:\n```json\n{}\n```\n'
    elif malformation=='bad_json': messages[0]['content']=messages[0]['content'].replace('"ui":','broken:')
    result=visible_state(messages)
    assert result['classifiable'] is False and result['candidate'] is False
    assert result['unknown_reason']
    assert 'sample' not in result['unknown_reason']


def test_format_failure_still_counts_visible_checkpoint_and_assigned_task(tmp_path):
    inputs=setup(tmp_path);make_failure(inputs,'not json','missing_action_json')
    result,_,_=analyze(inputs)
    assert result['assigned_task_denominator']==24
    assert result['counts']['tasks_with_observed_candidate']==24
    assert result['counts']['typed_format_failure_calls']==1


def test_unparseable_prompt_stays_in_coverage_denominator(tmp_path):
    inputs=setup(tmp_path);calls=read_rows(inputs['paths']['calls'])
    calls[0]['messages']=[{'role':'user','content':'No current observation anchor; thermometer only in history.'}]
    calls[0]['prompt_sha256']=digest_text(json.dumps(calls[0]['messages'],ensure_ascii=False,separators=(',',':')))
    write_rows(inputs['paths']['calls'],calls)
    result,_,_=analyze(inputs)
    assert result['assigned_task_denominator']==24
    assert result['counts']['tasks_with_unclassifiable_observation']==1
    assert result['counts']['tasks_with_observed_candidate']==23
    assert result['unknown_is_not_no_candidate'] is True


def test_generated_use_proposals_require_distinct_visible_uuid_binding(tmp_path):
    inputs=setup(tmp_path);calls=read_rows(inputs['paths']['calls'])
    outputs=['{"action":"USE","arg1":1,"arg2":2}',
             '{"action":"USE","arg1":1,"arg2":999}',
             '{"action":"USE","arg1":1,"arg2":1}',
             '{"action":"USE","arg1":2,"arg2":1}',
             '{"action":"SUBMIT","arg1":"Task completed!"}']
    for call,text in zip(calls,outputs):
        call.update(output_text=text,output_tokens=len(text),output_token_ids=list(map(ord,text)),controller_output_sha256=digest_text(text))
    write_rows(inputs['paths']['calls'],calls)
    for kind in ('outcomes','records'):
        rows=read_rows(inputs['paths'][kind])
        for row,call in zip(rows,calls): row['metadata'][f'hf_torch.{MODEL}.completion_tokens']=call['output_tokens']
        write_rows(inputs['paths'][kind],rows)
    result,_,_=analyze(inputs)
    assert result['counts']['valid_use_proposals']==4
    assert result['counts']['visible_bound_use_proposals']==2
    assert result['counts']['recognized_instrument_use_proposals']==1
    assert result['counts']['tasks_with_observed_candidate']==24


def test_terminal_labels_cannot_change_census(tmp_path):
    inputs=setup(tmp_path);result,audit,manifest_sha=analyze(inputs)
    audit['per_task_audited_summary']=[{'official_progress_score':1,'official_completed_successfully':True}]*24
    audit['primary_official_progress_score']={'mean':1.0}
    again=census(inputs['manifest'],[inputs['paths']['calls']],audit,inputs['source'],manifest_sha256=manifest_sha)
    assert result==again


@pytest.mark.parametrize('tamper', ['qualification','failed_exit','different_manifest','different_calls','source','analyzer'])
def test_census_does_not_bypass_terminal_provenance_gate(tmp_path,tamper):
    inputs=setup(tmp_path);_,audit,manifest_sha=analyze(inputs)
    if tamper=='qualification': audit['audit_scope']='runtime_qualification'
    elif tamper=='failed_exit': audit['slurm_accounting']['state']='FAILED'
    elif tamper=='different_manifest': audit['manifest_sha256']='other'
    elif tamper=='different_calls': inputs['paths']['calls'].write_text(inputs['paths']['calls'].read_text()+'\n')
    elif tamper=='source':
        path=inputs['source']/SOURCE_FILES[0];path.write_text(path.read_text().replace('Current Environment Observation:','New Heading:'))
        inputs['manifest']['visible_census_contract']['source_sha256'][SOURCE_FILES[0]]=sha256(path)
    elif tamper=='analyzer': inputs['manifest']['visible_census_contract']['analyzer_sha256']='other'
    with pytest.raises(ValueError): census(inputs['manifest'],[inputs['paths']['calls']],audit,inputs['source'],manifest_sha256=manifest_sha)


@pytest.mark.parametrize('first_unknown',[False,True])
def test_first_classifiable_candidate_is_online_and_earlier_unknowns_are_disclosed(tmp_path,first_unknown):
    inputs=setup(tmp_path);calls=read_rows(inputs['paths']['calls']);task=calls[0]['task_id']
    second=copy.deepcopy(calls[0])
    calls[0]['messages']=([{'role':'user','content':'No reliable current observation.'}] if first_unknown else
                          prompt([{'uuid':9,'name':'chair'}]))
    calls[0]['prompt_sha256']=digest_text(json.dumps(calls[0]['messages'],ensure_ascii=False,separators=(',',':')))
    calls.insert(1,second);write_rows(inputs['paths']['calls'],calls)
    for kind in ('records','outcomes'):
        rows=read_rows(inputs['paths'][kind]);m=rows[0]['metadata'];m[f'hf_torch.{MODEL}.calls']=2
        m[f'hf_torch.{MODEL}.prompt_tokens']=6;m[f'hf_torch.{MODEL}.completion_tokens']*=2;m['num_steps']=2
        write_rows(inputs['paths'][kind],rows)
    result,_,_=analyze(inputs)
    selection=result['per_task'][0]['first_classifiable_candidate']
    assert result['per_task'][0]['task_id']==task
    assert selection['task_call_index']==1
    assert selection['earlier_unclassifiable_calls']==int(first_unknown)
    assert selection['selection_uses_terminal_outcomes'] is False
