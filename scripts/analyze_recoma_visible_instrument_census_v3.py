#!/usr/bin/env python3
"""Source-bound visible instrument census after a complete v3 record audit.

Reads no all_data, terminal scorecards, success labels or hidden simulator state.
Candidate instrument/object visibility and USE proposals are descriptive, not
legal-action, useful-measurement, causal-opportunity or efficacy observations.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

if __package__:
    from .audit_recoma_discoveryworld_v3 import (classify_output, digest_text, expected_panel,
                                               require, sha256, source_contract)
else:
    from audit_recoma_discoveryworld_v3 import (classify_output, digest_text, expected_panel,
                                              require, sha256, source_contract)

VERSION = 'visible_instrument_census_v3'
INSTRUMENTS = ('microscope', 'ph meter', 'radiation meter', 'spectrometer',
               'thermometer', 'densitometer', 'proteomics meter')
SOURCE_FILES = ('agents/recoma/prompts/react_prompt.txt', 'agents/recoma/discoveryworld_promptlm.py',
                'discoveryworld/UserInterface.py', 'discoveryworld/objects/ScienceTools.py')
OBSERVATION_ANCHOR = 'Current Environment Observation:'
ACTIONS_ANCHOR = 'Valid Actions:'
BOUNDARY = ('Visible instrument/object candidate states and generated USE proposals only; '
            'not proven legal pairs, successful measurements, causal opportunities, '
            'intervention value, hidden CoT or efficacy.')


def checked_source(manifest, source_root):
    contract = manifest.get('visible_census_contract', {})
    require(contract.get('version') == VERSION, 'prospectively frozen visible census contract missing')
    require(contract.get('selection_uses_visible_fields_only') is True
            and contract.get('selection_uses_terminal_outcomes') is False,
            'visible census selection contract must exclude outcomes')
    require(contract.get('recognized_instrument_names') == list(INSTRUMENTS)
            and contract.get('checkpoint_rule') == 'first_classifiable_candidate_per_task',
            'visible census selection rule differs from prospective freeze')
    require(contract.get('analyzer_sha256') == sha256(Path(__file__)), 'census analyzer differs from freeze')
    pins = contract.get('source_sha256', {})
    require(set(pins) == set(SOURCE_FILES), 'all four census source pins required')
    contents = {}
    for relative in SOURCE_FILES:
        path = Path(source_root) / relative
        require(path.is_file() and sha256(path) == pins[relative], f'census source hash mismatch: {relative}')
        contents[relative] = path.read_text()
    template = contents[SOURCE_FILES[0]]
    require(template.count(OBSERVATION_ANCHOR) == 1 and template.count(ACTIONS_ANCHOR) == 1
            and OBSERVATION_ANCHOR+'\n```json\n{{ observation }}\n```' in template
            and ACTIONS_ANCHOR+'\n```json\n{{ known_actions}}\n```' in template,
            'pinned prompt observation/action anchors changed')
    adapter = ast.parse(contents[SOURCE_FILES[1]])
    methods = [n for n in ast.walk(adapter) if isinstance(n, ast.FunctionDef)
               and n.name == 'populate_template_dictionary']
    require(len(methods) == 1, 'pinned current-observation adapter missing')
    expressions = [n for n in ast.walk(methods[0]) if isinstance(n, ast.Call)]
    require(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'getAgentObservation'
                for n in expressions), 'adapter no longer takes current visible observation')
    require(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'deepcopy'
                and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == 'observation'
                for n in expressions), 'adapter observation copy contract changed')
    observed = [n.value for n in ast.walk(methods[0]) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == 'param_dict' and isinstance(t.slice, ast.Constant)
                        and t.slice.value == 'observation' for t in n.targets)]
    require(len(observed) == 1 and isinstance(observed[0], ast.Call)
            and isinstance(observed[0].func, ast.Attribute) and observed[0].func.attr == 'dumps'
            and isinstance(observed[0].args[0], ast.Name) and observed[0].args[0].id == 'observationNoVision',
            'prompt observation no longer serializes the current visible copy')
    ui = ast.parse(contents[SOURCE_FILES[2]])
    renderers = [n for n in ast.walk(ui) if isinstance(n, ast.FunctionDef)
                 and n.name == 'renderObjectSelectionBoxJSON']
    require(len(renderers) == 1, 'official visible object renderer missing')
    dictionaries = [n for n in ast.walk(renderers[0]) if isinstance(n, ast.Dict)]
    mappings = [{k.value:ast.dump(v) for k,v in zip(n.keys,n.values) if isinstance(k,ast.Constant)}
                for n in dictionaries]
    require(any(d.get('inventoryObjects') == ast.dump(ast.Name(id='invOut',ctx=ast.Load()))
                and d.get('accessibleEnvironmentObjects') == ast.dump(ast.Name(id='envOut',ctx=ast.Load()))
                for d in mappings), 'inventory/accessibility source mapping changed')
    require(any(d.get('name') == ast.dump(ast.parse('obj.name',mode='eval').body)
                and d.get('uuid') == ast.dump(ast.parse('obj.uuid',mode='eval').body) for d in mappings),
            'visible object name/UUID source mapping changed')
    instruments = ast.parse(contents[SOURCE_FILES[3]])
    names = {n.args[3].value.casefold() for n in ast.walk(instruments) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
             and n.func.value.id == 'Object' and n.func.attr == '__init__' and len(n.args)>=4
             and isinstance(n.args[3],ast.Constant) and isinstance(n.args[3].value,str)}
    require(set(INSTRUMENTS).issubset(names), 'recognized instrument names are not source-grounded')
    return pins


def anchored_json(contents, heading):
    pattern = re.compile(r'(?m)^'+re.escape(heading)+r'\n```json\n(.*?)\n```', re.DOTALL)
    matches = [m.group(1) for content in contents for m in pattern.finditer(content)]
    # Refuse ambiguous extra headings; never choose a favorable or historical match.
    anchors = sum(len(re.findall(r'(?m)^'+re.escape(heading)+r'$', content)) for content in contents)
    if anchors != 1 or len(matches) != 1:
        raise ValueError('missing_or_ambiguous_'+heading.lower().replace(' ','_').replace(':',''))
    return json.loads(matches[0])


def visible_state(messages):
    try:
        require(isinstance(messages,list) and messages, 'missing_prompt_messages')
        require(all(isinstance(m,dict) and m.get('role') in ('user','assistant','system')
                    and isinstance(m.get('content'),str) for m in messages), 'invalid_prompt_message')
        contents = [m['content'] for m in messages]
        observation = anchored_json(contents,OBSERVATION_ANCHOR)
        require(isinstance(observation,dict) and isinstance(observation.get('ui'),dict), 'invalid_current_ui')
        ui = observation['ui']
        dialog = ui.get('dialog_box',{}).get('is_in_dialog')
        require(type(dialog) is bool, 'unknown_dialog_status')
        objects, inventory = {}, set()
        for field in ('inventoryObjects','accessibleEnvironmentObjects'):
            entries=ui.get(field)
            require(isinstance(entries,list), 'missing_visible_object_list')
            for obj in entries:
                require(isinstance(obj,dict) and type(obj.get('uuid')) is int and obj['uuid']>=0
                        and isinstance(obj.get('name'),str) and bool(obj['name']), 'invalid_visible_object_identity')
                uid,name=obj['uuid'],obj['name']
                require(uid not in objects or objects[uid] == name, 'conflicting_visible_object_identity')
                objects[uid]=name
                if field=='inventoryObjects': inventory.add(uid)
        if dialog:
            use_known=False
        else:
            actions=anchored_json(contents,ACTIONS_ANCHOR)
            require(isinstance(actions,dict), 'invalid_known_action_set')
            use_known='USE' in actions
        sensors={uid for uid,name in objects.items() if name.casefold() in INSTRUMENTS}
        candidate=not dialog and use_known and bool(sensors) and len(objects)>1
        return {'classifiable':True,'unknown_reason':None,'candidate':candidate,
                'objects':objects,'sensors':sensors,'inventory':inventory,'in_dialog':dialog,'use_known':use_known}
    except (ValueError,TypeError,KeyError,AttributeError,json.JSONDecodeError) as exc:
        reason=str(exc) if type(exc) is ValueError else 'invalid_current_observation_json'
        # Errors never expose raw observations or parser snippets in the public census.
        if len(reason)>100 or '\n' in reason: reason='invalid_current_observation_json'
        return {'classifiable':False,'unknown_reason':reason,'candidate':False,
                'objects':{},'sensors':set(),'inventory':set(),'in_dialog':None,'use_known':False}


def census(manifest,call_paths,record_audit,source_root,*,manifest_sha256):
    require(record_audit.get('audit_status')=='PASS' and record_audit.get('audit_scope') in ('full_panel','balanced_census'),
            'complete terminal production record audit required before census')
    require(record_audit.get('manifest_sha256')==manifest_sha256,'census manifest differs from terminal audit')
    require(record_audit.get('slurm_accounting',{}).get('state')=='COMPLETED'
            and record_audit['slurm_accounting'].get('exit_code')=='0:0','terminal allocation was not successful')
    require(record_audit.get('auditor_sha256')==manifest.get('artifact_sha256',{}).get('audit_recoma_discoveryworld_v3.py'),
            'terminal auditor identity differs from frozen manifest')
    pins=checked_source(manifest,source_root)
    scenarios,_=source_contract(manifest,source_root)
    scope,expected=expected_panel(manifest,scenarios)
    require(scope==record_audit['audit_scope'] and record_audit.get('n_tasks')==len(expected),
            'terminal audited denominator/scope differs')
    grouped=defaultdict(list); call_hashes={}
    for path in call_paths:
        path=Path(path); call_hashes[str(path)]=sha256(path)
        for line in path.read_text().splitlines():
            if line.strip():
                call=json.loads(line)
                require(isinstance(call,dict) and call.get('task_id') in expected and call.get('status')=='ok',
                        'non-scientific/unknown call entered census')
                grouped[call['task_id']].append(call)
    require(call_hashes==record_audit.get('input_sha256',{}).get('calls'),
            'census call files differ from independently audited inputs')
    require(set(grouped)==set(expected),'census does not retain entire assigned denominator')
    totals=Counter(); unknowns=Counter(); tasks=[]; strata=defaultdict(list)
    for task,assigned in expected.items():
        summary={'task_id':task,'logged_decision_calls':len(grouped[task]),'evaluable_calls':0,
                 'unclassifiable_calls':0,'candidate_state_calls':0,'valid_use_proposals':0,
                 'visible_bound_use_proposals':0,'recognized_instrument_use_proposals':0,
                 'thought_bearing_calls':0,'typed_format_failure_calls':0,'first_classifiable_candidate':None}
        for index,call in enumerate(grouped[task]):
            view=visible_state(call.get('messages'))
            if view['classifiable']:
                summary['evaluable_calls']+=1
            else:
                summary['unclassifiable_calls']+=1; unknowns[view['unknown_reason']]+=1
            if view['candidate']:
                summary['candidate_state_calls']+=1
                if summary['first_classifiable_candidate'] is None:
                    summary['first_classifiable_candidate']={'task_call_index':index,'prompt_sha256':call['prompt_sha256'],
                        'earlier_unclassifiable_calls':summary['unclassifiable_calls'],
                        'selection_uses_terminal_outcomes':False}
            failure,action=classify_output(call['output_text'])
            if failure:
                summary['typed_format_failure_calls']+=1
            else:
                summary['thought_bearing_calls']+=int(isinstance(action.get('thought'),str) and bool(action['thought'].strip()))
                if action.get('action')=='USE':
                    summary['valid_use_proposals']+=1
                    a,b=action.get('arg1'),action.get('arg2')
                    bound=type(a) is int and type(b) is int and a!=b and a in view['objects'] and b in view['objects']
                    summary['visible_bound_use_proposals']+=int(bound)
                    summary['recognized_instrument_use_proposals']+=int(bound and a in view['sensors'])
        summary['has_evaluable_observation']=summary['evaluable_calls']>0
        summary['has_unclassifiable_observation']=summary['unclassifiable_calls']>0
        summary['has_observed_candidate']=summary['first_classifiable_candidate'] is not None
        totals.update({k:v for k,v in summary.items() if type(v) is int})
        totals['tasks_with_evaluable_observation']+=int(summary['has_evaluable_observation'])
        totals['tasks_with_unclassifiable_observation']+=int(summary['has_unclassifiable_observation'])
        totals['tasks_with_observed_candidate']+=int(summary['has_observed_candidate'])
        key=(assigned.get('scenario_name',assigned.get('scenario')),assigned['difficulty'])
        strata[key].append(summary); tasks.append(summary)
    n=len(expected)
    rates=[sum(row['has_observed_candidate'] for row in rows)/len(rows) for rows in strata.values()]
    return {'status':'PASS_VISIBLE_ONLY_CENSUS','version':VERSION,'audit_scope':scope,'assigned_task_denominator':n,
            'independent_template_difficulty_strata':len(strata),'recognized_instrument_names':list(INSTRUMENTS),
            'counts':dict(totals),'unknown_reasons':dict(unknowns),
            'observed_candidate_task_fraction':totals['tasks_with_observed_candidate']/n,
            'equal_stratum_observed_candidate_fraction':sum(rates)/len(rates),
            'unknown_is_not_no_candidate':True,'claim_boundary':BOUNDARY,
            'first_checkpoint_rule':'First classifiable candidate in each task call order; unknown earlier calls disclosed.',
            'uses_terminal_scorecards_or_success':False,'new_counterfactual_interventions':0,
            'per_task':tasks,'per_stratum':[{'scenario':key[0],'difficulty':key[1],'assigned_tasks':len(rows),
                'tasks_with_evaluable_observation':sum(x['has_evaluable_observation'] for x in rows),
                'tasks_with_unclassifiable_observation':sum(x['has_unclassifiable_observation'] for x in rows),
                'tasks_with_observed_candidate':sum(x['has_observed_candidate'] for x in rows)} for key,rows in strata.items()],
            'source_sha256':pins,'call_file_sha256':call_hashes,'manifest_sha256':manifest_sha256,
            'analyzer_sha256':sha256(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','record-audit','source-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--call-files',nargs='+',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'refuse to overwrite prior visible census')
    result=census(json.loads(args.manifest.read_text()),args.call_files,json.loads(args.record_audit.read_text()),
                  args.source_root,manifest_sha256=sha256(args.manifest))
    result['record_audit_sha256']=sha256(args.record_audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','assigned_task_denominator','observed_candidate_task_fraction')},sort_keys=True))


if __name__=='__main__': main()
