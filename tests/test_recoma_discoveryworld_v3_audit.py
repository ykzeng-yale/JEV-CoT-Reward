"""Adversarial record/source audit tests; no simulator, weights, or API calls."""
import copy
import ast
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_recoma_discoveryworld_v3 import (SOURCE_FILES, SERIALIZATION_SOURCE_FILES, audit,
                 classify_output, digest_text, sha256, source_contract, reconstruct_role_messages)

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
SCENARIOS = ['Combinatorial Chemistry', 'Archaeology Dating', 'Plant Nutrients', 'Reactor Lab',
             'Lost in Translation', 'Space Sick', 'Proteomics', "It's (not) Rocket Science!"]
CONTRACT = 'recoma_scientific_failure_accounting_v3'
ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT/'runs/recoma-runtime-v3-delivery-control/source-v3'


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return types.SimpleNamespace(shape=(1, 3))

    def decode(self, ids, **kwargs):
        return ''.join(chr(value) for value in ids)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_source(tmp_path):
    source = tmp_path / 'source/discoveryworld'
    official_task = '''class Task:
    def getScoreNormalized(self):
        return self.score / self.maxScore
    def taskProgressDict(self):
        out = {}
        out["score"] = self.score
        out["maxScore"] = self.maxScore
        out["scoreNormalized"] = self.getScoreNormalized()
        out["completed"] = self.completed
        out["completedSuccessfully"] = self.completedSuccessfully
        return out
'''
    contents = {
        SOURCE_FILES[0]: official_task,
        SOURCE_FILES[1]: 'class API:\n    def getTaskScorecard(self):\n        return self.ui.getFullTaskProgressJSON()\n',
        SOURCE_FILES[2]: 'class UI:\n    def getFullTaskProgressJSON(self):\n        return [task.taskProgressDict() for task in self.tasks]\n',
        SOURCE_FILES[3]: 'SCENARIO_INFOS = ' + repr({name: {'difficulty': ['Easy', 'Normal', 'Challenge'],
                                                       'variations': ['1', '2', '3', '4', '5']}
                                                  for name in SCENARIOS}) + '\n',
        SOURCE_FILES[4]: 'class Controller:\n    def __init__(self):\n        self.full_code_regex = r"```json\\n(.*?)```"\n        self.partial_code_regex = r".*```json\\n(.*)"\n',
    }
    for relative, content in contents.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    contract = source.parent / 'recoma/recoma/utils/task_accounting.py'
    contract.parent.mkdir(parents=True)
    contract.write_text('# Frozen test scientific-failure runtime source.\n')
    for relative in SERIALIZATION_SOURCE_FILES:
        path=source.parent/relative; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((DELIVERY/relative).read_bytes())
    inventory=source.parent/'SHA256SUMS'
    inventory.write_text(''.join(f'{sha256(source.parent/relative)}  {relative}\n'
                                 for relative in SERIALIZATION_SOURCE_FILES))
    return source, {relative: sha256(source / relative) for relative in SOURCE_FILES}, sha256(contract), sha256(inventory)


def fixture(tmp_path, scope='runtime_qualification'):
    source, pins, contract_hash, source_manifest_hash = make_source(tmp_path)
    inventory = [{'scenario_name': s, 'difficulty': d, 'random_seed': seed}
                 for s in SCENARIOS for d in ['Easy', 'Normal', 'Challenge'] for seed in range(5)]
    if scope == 'runtime_qualification':
        inventory = [{'scenario_name': SCENARIOS[0], 'difficulty': 'Easy', 'random_seed': 5},
                     {'scenario_name': SCENARIOS[-1], 'difficulty': 'Challenge', 'random_seed': 5}]
    ids = [f"{x['scenario_name']}_{x['difficulty']}_{x['random_seed']}" for x in inventory]
    manifest = {'audit_scope': scope, 'qualification_only': scope == 'runtime_qualification',
                'task_instances': inventory, 'seeds': [5] if scope == 'runtime_qualification' else list(range(5)),
                'endpoint_source_sha256': pins, 'runtime_contract': CONTRACT,
                'source_manifest_sha256':source_manifest_hash,
                'runtime_contract_sha256': contract_hash, 'model': MODEL, 'model_revision': REVISION,
                'max_new_tokens_per_generation': 400, 'temperature': 0.0, 'stop_sequences': ['```\n'],
                'max_environment_actions_per_episode': 800, 'max_llm_calls_per_episode': 1620,
                'resources': {'account': 'pi_fl426', 'gpu_count': 1, 'cpus': 4, 'memory_gib': 64,
                              'partitions': ['gpu_h200', 'gpu_b200'], 'wall_minutes': 120},
                'runtime': {'torch_version': '2.9.1', 'transformers_version': '4.55.2'},
                'workers': [{'task_ids': ids}], 'qualified_context_token_ceiling': 32768, 'context_stress_input_tokens': 32768,
                'context_stress_output_tokens': 400, 'claim_boundary': 'Runtime qualification only'}
    messages = [{'role': 'user', 'content': 'Only agent-visible task and observation.'}]
    text = '  {"action":"WAIT","thought":"Check visible state"}'
    calls, records, outcomes, events = [], [], [], []
    prefix = f'hf_torch.{MODEL}'
    for i, (row, task) in enumerate(zip(inventory, ids)):
        calls.append({'status': 'ok', 'task_id': task, 'model': MODEL, 'model_revision': REVISION,
                      'messages': messages, 'prompt_sha256': digest_text(json.dumps(messages, ensure_ascii=False,
                                                                                separators=(',', ':'))),
                      'input_tokens': 3, 'output_tokens': len(text), 'output_token_ids': list(map(ord, text)),
                      'output_text': text, 'controller_output_sha256': digest_text(text.lstrip()),
                      'temperature': 0.0, 'max_new_tokens': 400, 'elapsed_seconds': 1.0,
                      'peak_allocated_gpu_bytes': 100, 'peak_reserved_gpu_bytes': 200})
        metadata = {'num_steps': 1, prefix+'.calls': 1, prefix+'.prompt_tokens': 3,
                    prefix+'.completion_tokens': len(text),
                    'final_scorecard': [{'taskName': 'Official task', 'score': 1, 'maxScore': 2,
                                         'scoreNormalized': 0.5, 'completed': False,
                                         'completedSuccessfully': False,
                                         'scoreCard': [{'score': 1, 'maxScore': 2, 'completed': False}]}]}
        record = {**row, 'task_id': task, 'predicted': 0.5, 'metadata': metadata,
                  'scientific_task_status': 'completed', 'failure_adjusted_completed_successfully': False,
                  'task_wall_seconds': 2.0, 'runtime_contract': CONTRACT, 'runtime_contract_sha256': contract_hash}
        records.append(record)
        # Actual upstream dump_predictions keeps a numeric JSON scalar string;
        # task_accounting's durable record parses it to a numeric scalar.
        outcomes.append({**row, 'task_id': task, 'predicted': '0.5', 'metadata': copy.deepcopy(metadata),
                         'correct': '0'})
        events += [{'event': 'task_started', 'task_id': task, 'monotonic_seconds': 2.0*i},
                   {'event': 'task_ended', 'task_id': task, 'monotonic_seconds': 2.0*i+1,
                    'scientific_task_status': 'completed'}]
    paths = {kind: tmp_path / (kind+'.jsonl') for kind in ('outcomes', 'calls', 'records', 'events', 'stress')}
    for key, rows in [('calls', calls), ('outcomes', outcomes), ('records', records), ('events', events)]:
        write_rows(paths[key], rows)
    stress_ids = [73] * 32768
    stress = {'status': 'completed', 'model': MODEL, 'model_revision': REVISION, 'input_tokens': 32768,
              'output_tokens': 400, 'input_token_ids': stress_ids, 'output_token_ids': [73]*400,
              'output_text': 'I'*400, 'input_token_ids_sha256': digest_text(json.dumps(stress_ids, separators=(',', ':'))),
              'max_new_tokens': 400, 'do_sample': False, 'elapsed_seconds': 2.0,
              'scientific_episode': False, 'model_calls': 1, 'jev_calls': 0, 'hosted_calls': 0,
              'output_token_ids_sha256': digest_text(json.dumps([73]*400, separators=(',', ':'))),
              'peak_allocated_gpu_bytes': 300, 'peak_reserved_gpu_bytes': 400}
    paths['stress'].write_text(json.dumps(stress, indent=2)+'\n')
    report = {'status': 'completed', 'performed_generation': True, 'worker_index': 0,
              'expected_task_ids': ids, 'model_revision': REVISION, 'model_path': '/cache/snapshots/'+REVISION,
              'torch': '2.9.1+cu128', 'transformers': '4.55.2', 'cuda_device_count': 1,
              'cuda_device': 'NVIDIA H200', 'max_memory_allocated_bytes': 300,
              'cuda_total_memory_bytes': 1000,
              'max_memory_reserved_bytes': 400, 'elapsed_seconds': 600.0}
    if scope != 'runtime_qualification':
        add_model_content_fixture(tmp_path, manifest, paths, report)
    if scope == 'runtime_qualification':
        report['context_stress'] = {k: v for k, v in stress.items()
                                    if k not in ('input_token_ids', 'output_token_ids', 'output_text')}
    resources = {'state': 'COMPLETED', 'exit_code': '0:0', 'account': 'pi_fl426', 'gpu_count': 1,
                 'cpu_count': 4, 'memory_gib': 64, 'partition': 'gpu_h200', 'qos': 'normal',
                 'model_revision': REVISION, 'elapsed_seconds': 600.0}
    return {'manifest': manifest, 'paths': paths, 'runtime': report, 'resources': resources, 'source': source}


def add_model_content_fixture(tmp_path, manifest, paths, report):
    """Small receipt fixture mirrors both the verifier's full inventory and CLI."""
    paths.update(model_checksums=tmp_path/'model_snapshot_SHA256SUMS',
                 model_verifier=tmp_path/'verify_model_snapshot_v3.py',
                 model_before=tmp_path/'model-verification-before.json',
                 model_after=tmp_path/'model-verification-after.json')
    content={'config.json': '{}', 'model.safetensors': 'frozen test weights'}
    expected={name:digest_text(value) for name,value in content.items()}
    paths['model_checksums'].write_text(''.join(f'{digest}  {name}\n' for name,digest in expected.items()))
    paths['model_verifier'].write_text('# Frozen verifier source fixture; real verifier separately fault-tested.\n')
    rows=[]
    for name, text in content.items():
        addressed=name=='model.safetensors'
        target='/cache/blobs/'+expected[name] if addressed else report['model_path']+'/'+name
        rows.append({'relative_path':name, 'sha256':expected[name], 'expected_sha256':expected[name],
                     'bytes':len(text), 'snapshot_reference_path':report['model_path']+'/'+name,
                     'resolved_content_path':target, 'reference_type':'symlink' if addressed else 'regular_file',
                     'declared_symlink_target':'../../blobs/'+expected[name] if addressed else None,
                     'content_addressed_blob_name':expected[name] if addressed else None,
                     'content_addressed_blob_type':'hf_lfs_sha256' if addressed else None,
                     'content_addressed_blob_digest':expected[name] if addressed else None,
                     'content_addressed_blob_verified':True if addressed else None})
    receipt={'status':'PASS_MODEL_SNAPSHOT_CONTENT','schema':'model_snapshot_content_verification_v3',
             'snapshot_revision':REVISION,'snapshot_path':report['model_path'],
             'resolved_snapshot_path':report['model_path'],
             'checksum_file_sha256':sha256(paths['model_checksums']),
             'verifier_sha256':sha256(paths['model_verifier']), 'files':rows, 'file_count':len(rows),
             'total_verified_reference_bytes':sum(x['bytes'] for x in rows),
             'unique_content_files':len(rows), 'unique_content_bytes':sum(x['bytes'] for x in rows),
             'hf_content_addressed_symlinks_verified':1,'non_content_addressed_symlinks':0,
             'elapsed_seconds':3.0,'process_cpu_seconds':2.0,'model_loaded':False,'model_imports':False,
             'downloaded_bytes':0,'model_generations':0,'hosted_calls':0,'jev_calls':0}
    for key in ('model_before','model_after'):
        paths[key].write_text(json.dumps(receipt,indent=2)+'\n')
    manifest.update(qualified_gpu={'name':report['cuda_device'],
                                  'total_memory_bytes':report['cuda_total_memory_bytes']},
                    model_content_contract={'snapshot_revision':REVISION,
                       'checksums_sha256':sha256(paths['model_checksums']),
                       'verifier_sha256':sha256(paths['model_verifier']),
                       'require_compute_allocation_pre_and_post_verification':True,
                       'verification_latency_is_allocated_cost':True})


def run(inputs):
    p = inputs['paths']
    return audit(inputs['manifest'], [p['outcomes']], [p['calls']], inputs['resources'], Path('/unused'),
                 task_record_paths=[p['records']], event_paths=[p['events']], source_root=inputs['source'],
                 runtime_report=inputs['runtime'], tokenizer=FakeTokenizer(),
                 stress_paths=[p['stress']] if inputs['manifest']['audit_scope']=='runtime_qualification' else None,
                 model_checksum_path=p.get('model_checksums'), model_verifier_path=p.get('model_verifier'),
                 model_verification_before=p.get('model_before'), model_verification_after=p.get('model_after'))


def edit_row(inputs, kind, field, value, index=0):
    if kind == 'stress':
        row=json.loads(inputs['paths'][kind].read_text()); row[field]=value
        inputs['paths'][kind].write_text(json.dumps(row,indent=2)+'\n')
    else:
        rows = read_rows(inputs['paths'][kind]); rows[index][field] = value
        write_rows(inputs['paths'][kind], rows)


def make_failure(inputs, text, code):
    calls = read_rows(inputs['paths']['calls'])
    calls[0].update(output_text=text, output_tokens=len(text), output_token_ids=list(map(ord, text)),
                    controller_output_sha256=digest_text(text.lstrip()))
    write_rows(inputs['paths']['calls'], calls)
    for kind in ('records', 'outcomes'):
        rows = read_rows(inputs['paths'][kind])
        meta = rows[0]['metadata']
        meta[f'hf_torch.{MODEL}.completion_tokens'] = len(text)
        meta['scientific_task_failure'] = {'code': code, 'output_sha256': digest_text(text.lstrip()),
                                          'retry_or_resample': False}
        meta['scientific_task_status'] = 'scientific_failure'
        if kind == 'records':
            rows[0]['scientific_task_status'] = 'scientific_failure'
        write_rows(inputs['paths'][kind], rows)
    edit_row(inputs, 'events', 'scientific_task_status', 'scientific_failure', index=1)


def test_qualification_checks_two_full_budget_runtime_tasks_and_charges_stress(tmp_path):
    inputs = fixture(tmp_path)
    result = run(inputs)
    assert result['audit_status'] == 'PASS' and result['n_tasks'] == 2
    assert result['actual_gpu_hours_from_slurm'] == pytest.approx(1/6)
    assert result['context_stress_prompt_tokens'] == 32768
    assert result['all_prompt_tokens_including_runtime_stress'] == 32774
    assert result['all_generated_tokens_including_failed_final_calls_and_runtime_stress'] == result['completion_tokens']+400
    assert 'primary_official_progress_score' not in result
    assert 'no efficacy rate/CI' in result['qualification_boundary']
    assert 'no independent world-state replay' in result['outcome_audit_boundary']
    inventory=result['runtime_source_inventory']
    assert inventory['path']==str(inputs['source'].parent/'SHA256SUMS')
    assert inventory['sha256']==sha256(inventory['path'])
    assert result['prompt_serialization_contract']['reconstructed_message_key_order']==['role','content']


def test_actual_generator_and_backend_logger_roundtrip_preserves_prompt_sha(tmp_path):
    """Execute exact frozen parsing/hash/log bytes without Torch or inference."""
    inputs=fixture(tmp_path)
    core_tree=ast.parse((DELIVERY/SERIALIZATION_SOURCE_FILES[0]).read_text())
    backend_tree=ast.parse((DELIVERY/SERIALIZATION_SOURCE_FILES[1]).read_text())
    extraction=next(n for n in ast.walk(core_tree) if isinstance(n,ast.FunctionDef)
                    and n.name=='extract_role_messages')
    logger=next(n for n in backend_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_append_jsonl')
    namespace={'json':json,'re':re,'Path':Path,'os':os}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extraction,logger],type_ignores=[])),
                 'actual-frozen-generator-and-log','exec'),namespace)
    messages=namespace['extract_role_messages'](None,'USER:\nObserve μ instrument.\nASSISTANT:\nPrior answer.\nUSER:\nAct now.')
    assert all(list(message)==['role','content'] for message in messages)
    prompt_assignment=next(n for n in ast.walk(backend_tree) if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='prompt_text' for t in n.targets))
    actual_prompt=eval(compile(ast.Expression(body=prompt_assignment.value), 'actual-backend-prompt-hash','eval'),
                       {'json':json,'messages':messages})
    calls=read_rows(inputs['paths']['calls'])
    calls[0]['messages']=messages;calls[0]['prompt_sha256']=digest_text(actual_prompt)
    logged=tmp_path/'actual-backend-sorted-calls.jsonl'
    for call in calls: namespace['_append_jsonl'](str(logged),call)
    durable=read_rows(logged)[0]
    assert all(list(message)==['content','role'] for message in durable['messages'])
    assert digest_text(json.dumps(durable['messages'],ensure_ascii=False,separators=(',',':'))) != durable['prompt_sha256']
    restored=reconstruct_role_messages(durable['messages'])
    assert restored==messages and digest_text(json.dumps(restored,ensure_ascii=False,separators=(',',':'))) == durable['prompt_sha256']
    inputs['paths']['calls']=logged
    assert run(inputs)['audit_status']=='PASS'


@pytest.mark.parametrize('mutation', ['role','content','extra','missing','nonstring'])
def test_message_order_restoration_does_not_drop_or_change_message_fields(tmp_path, mutation):
    inputs=fixture(tmp_path);rows=read_rows(inputs['paths']['calls']);message=rows[0]['messages'][0]
    if mutation=='role':message['role']='assistant'
    elif mutation=='content':message['content']='altered visible prompt'
    elif mutation=='extra':message['hidden_field']='not part of generator contract'
    elif mutation=='missing':message.pop('role')
    else:message['role']=1
    write_rows(inputs['paths']['calls'],rows)
    with pytest.raises(ValueError,match='prompt hash mismatch|role/content message schema'):
        run(inputs)


@pytest.mark.parametrize('target', ['inventory','generator','backend','logger_semantics'])
def test_prompt_reconstruction_requires_exact_frozen_source_and_logger_semantics(tmp_path,target):
    inputs=fixture(tmp_path);bundle=inputs['source'].parent;inventory=bundle/'SHA256SUMS'
    if target=='inventory':inventory.write_text(inventory.read_text()+'# changed\n')
    elif target in ('generator','backend'):
        path=bundle/SERIALIZATION_SOURCE_FILES[0 if target=='generator' else 1]
        path.write_text(path.read_text()+'\n# changed\n')
    else:
        path=bundle/SERIALIZATION_SOURCE_FILES[1]
        path.write_text(path.read_text().replace('json.dumps(row, sort_keys=True)','json.dumps(row, sort_keys=False)'))
        inventory.write_text(''.join(f'{sha256(bundle/relative)}  {relative}\n'
                                      for relative in SERIALIZATION_SOURCE_FILES))
        inputs['manifest']['source_manifest_sha256']=sha256(inventory)
    with pytest.raises(ValueError,match='source inventory hash|serialization source hash|recursive-key-sort'):
        run(inputs)


def test_full_production_retains_every_task_and_stratum(tmp_path):
    inputs = fixture(tmp_path, 'full_panel')
    result = run(inputs)
    assert result['n_tasks'] == 120 and result['independent_strata_count'] == 24
    assert result['primary_official_progress_score']['mean'] == 0.5
    assert len(result['per_task_audited_summary']) == 120
    assert result['primary_official_progress_score']['population_confidence_interval'] is False
    verification=result['model_content_verification']
    assert verification['status']=='PASS_PRE_POST_FROZEN_MODEL_CONTENT'
    assert verification['verification_elapsed_seconds']==6.0
    assert verification['verification_process_cpu_seconds']==4.0
    assert verification['latency_included_in_allocation_cost'] is True
    assert verification['file_count']==2
    assert verification['input_paths'] == {
        'checksums':str(inputs['paths']['model_checksums']),
        'verifier':str(inputs['paths']['model_verifier']),
        'before':str(inputs['paths']['model_before']), 'after':str(inputs['paths']['model_after'])}
    assert verification['input_sha256'] == {key:sha256(path) for key,path in verification['input_paths'].items()}


@pytest.mark.parametrize('kind', ['model_before', 'model_after', 'model_verifier', 'model_checksums'])
def test_production_requires_actual_frozen_artifacts_and_both_content_receipts(tmp_path, kind):
    inputs=fixture(tmp_path,'full_panel'); inputs['paths'][kind].unlink()
    with pytest.raises(ValueError,match='frozen model artifacts and both model-content receipts'):
        run(inputs)


@pytest.mark.parametrize('kind', ['model_verifier', 'model_checksums'])
def test_changed_frozen_model_artifact_is_not_excused_by_receipt_pass(tmp_path, kind):
    inputs=fixture(tmp_path,'full_panel')
    path=inputs['paths'][kind]; path.write_text(path.read_text()+'# changed\n')
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        run(inputs)


@pytest.mark.parametrize('field,value', [
    ('status','FAIL_MODEL_SNAPSHOT_CONTENT'), ('snapshot_revision','0'*40),
    ('verifier_sha256','0'*64), ('checksum_file_sha256','0'*64),
    ('resolved_snapshot_path','/wrong/cache/'+REVISION), ('model_generations',1),
    ('model_loaded',True), ('elapsed_seconds',601.0), ('file_count',1),
    ('total_verified_reference_bytes',1), ('hf_content_addressed_symlinks_verified',0),
])
def test_wrong_post_verification_receipt_blocks_production_claims(tmp_path, field, value):
    inputs=fixture(tmp_path,'full_panel'); path=inputs['paths']['model_after']
    receipt=json.loads(path.read_text()); receipt[field]=value
    path.write_text(json.dumps(receipt)+'\n')
    with pytest.raises(ValueError): run(inputs)


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'hash', 'hf_target', 'hf_false', 'bytes'])
def test_full_content_inventory_is_checked_beyond_selfreported_pass(tmp_path, change):
    inputs=fixture(tmp_path,'full_panel'); path=inputs['paths']['model_after']
    receipt=json.loads(path.read_text()); rows=receipt['files']
    if change=='missing': rows.pop()
    elif change=='extra': rows.append({**rows[0],'relative_path':'unexpected.bin'})
    elif change=='duplicate': rows.append(copy.deepcopy(rows[0]))
    elif change=='hash': rows[-1]['sha256']=rows[-1]['expected_sha256']='0'*64
    elif change=='hf_target': rows[-1]['resolved_content_path']='/cache/blobs/'+'0'*64
    elif change=='hf_false': rows[-1]['content_addressed_blob_verified']=False
    elif change=='bytes': rows[-1]['bytes']+=1; receipt['total_verified_reference_bytes']+=1; receipt['unique_content_bytes']+=1
    receipt['file_count']=len(rows); path.write_text(json.dumps(receipt)+'\n')
    with pytest.raises(ValueError): run(inputs)


@pytest.mark.parametrize('field,value,error', [
    ('cuda_device','NVIDIA A40','measured qualified'),
    ('cuda_total_memory_bytes',900,'measured qualified'),
    ('cuda_total_memory_bytes',None,'physical CUDA memory'),
    ('max_memory_reserved_bytes',1001,'physical capacity'),
])
def test_production_requires_exact_qualified_gpu_and_physical_memory(tmp_path, field, value, error):
    inputs=fixture(tmp_path,'full_panel'); inputs['runtime'][field]=value
    with pytest.raises(ValueError,match=error): run(inputs)


def test_runtime_qualification_keeps_original_content_receipt_boundary(tmp_path):
    inputs=fixture(tmp_path)
    inputs['runtime'].pop('cuda_total_memory_bytes')
    assert run(inputs)['audit_status']=='PASS'


@pytest.mark.parametrize('text,code', [('not json', 'missing_action_json'),
    ('```json\n{broken', 'invalid_action_json'), ('  []', 'non_object_action_json'),
    ('{"action":"SUBMIT","arg1":3}', 'invalid_submit_arguments')])
def test_typed_final_failure_is_scientific_and_partial_score_stays(tmp_path, text, code):
    inputs = fixture(tmp_path, 'full_panel'); make_failure(inputs, text, code)
    result = run(inputs)
    assert result['scientific_failure_count'] == 1
    assert result['primary_official_progress_score']['mean'] == 0.5
    assert result['per_task_audited_summary'][0]['official_progress_score'] == 0.5
    assert result['n_tasks'] == 120
    assert result['all_generated_tokens_including_failed_final_calls_and_runtime_stress'] == sum(
        x['output_tokens'] for x in read_rows(inputs['paths']['calls']))


def test_valid_but_denied_json_is_not_a_scientific_format_failure(tmp_path):
    inputs = fixture(tmp_path)
    text = '{"action":"NOT_AN_OFFICIAL_ACTION"}'
    calls = read_rows(inputs['paths']['calls']); calls[0].update(output_text=text, output_tokens=len(text),
        output_token_ids=list(map(ord,text)),controller_output_sha256=digest_text(text))
    write_rows(inputs['paths']['calls'], calls)
    for kind in ('records','outcomes'):
        rows=read_rows(inputs['paths'][kind]); rows[0]['metadata'][f'hf_torch.{MODEL}.completion_tokens']=len(text)
        write_rows(inputs['paths'][kind],rows)
    assert run(inputs)['scientific_failure_count'] == 0


@pytest.mark.parametrize('text,expected', [
    ('```json\n{"action":"WAIT"}``` extra ```json\nbad```', None),
    ('prefix ```json\n{"action":"WAIT"}', None),
    ('```json\n```', 'invalid_action_json'),
    ('```json\n', 'missing_action_json'),
    ('"valid JSON string"', 'non_object_action_json'),
    ('{"action":"SUBMIT","arg1":"answer","thought":false}', 'invalid_submit_arguments'),
])
def test_frozen_parser_boundary(text, expected):
    assert classify_output(text)[0] == expected


@pytest.mark.parametrize('field,value,error', [
    ('model_revision', 'other', 'revision mismatch'), ('output_tokens', 1, 'completion token count'),
    ('output_text', '{}', 'controller output identity'), ('controller_output_sha256', 'bad', 'controller output identity'),
    ('input_tokens', 4, 'prompt token count'), ('status', 'error', 'infrastructure/model-call failure'),
    ('peak_allocated_gpu_bytes', -1, 'allocated memory'), ('temperature', 1.0, 'generation contract'),
])
def test_bad_call_records_rejected(tmp_path, field, value, error):
    inputs=fixture(tmp_path); edit_row(inputs,'calls',field,value)
    with pytest.raises(ValueError, match=error): run(inputs)


def test_selfreported_normalized_score_is_independently_recomputed(tmp_path):
    inputs=fixture(tmp_path)
    for kind in ('outcomes','records'):
        rows=read_rows(inputs['paths'][kind]); rows[0]['metadata']['final_scorecard'][0]['scoreNormalized']=0.9
        rows[0]['predicted']=0.9; write_rows(inputs['paths'][kind],rows)
    with pytest.raises(ValueError,match='independently recomputed'): run(inputs)


@pytest.mark.parametrize('value', ['0.5', '5e-1', 0.5])
def test_numeric_serialization_equivalence_requires_official_endpoint(tmp_path, value):
    inputs=fixture(tmp_path); edit_row(inputs, 'outcomes', 'predicted', value)
    assert run(inputs)['qualification_task_audit'][0]['official_progress_score'] == 0.5


@pytest.mark.parametrize('kind', ['records', 'outcomes'])
@pytest.mark.parametrize('value', ['0.50001', 'bad', 'NaN', float('inf'), True, [], {'score': 0.5}])
def test_numeric_serialization_does_not_excuse_malformed_or_wrong_endpoint(tmp_path, kind, value):
    inputs=fixture(tmp_path); edit_row(inputs, kind, 'predicted', value)
    with pytest.raises(ValueError, match='answerer prediction'):
        run(inputs)


def test_generic_string_exact_match_is_not_a_discoveryworld_endpoint(tmp_path):
    inputs=fixture(tmp_path, 'full_panel')
    before=run(inputs)
    for value in ('0', '1', 'malformed irrelevant generic EM'):
        rows=read_rows(inputs['paths']['outcomes'])
        for row in rows:
            row['correct']=value
        write_rows(inputs['paths']['outcomes'], rows)
        after=run(inputs)
        for field in ('primary_official_progress_score', 'official_completed_successfully_rate',
                      'failure_adjusted_completed_successfully_rate', 'per_task_audited_summary'):
            assert after[field] == before[field]


def test_failure_hash_or_retry_is_rejected(tmp_path):
    inputs=fixture(tmp_path); make_failure(inputs,'  not json','missing_action_json')
    for kind in ('records','outcomes'):
        rows=read_rows(inputs['paths'][kind]); rows[0]['metadata']['scientific_task_failure']['output_sha256']=digest_text('  not json')
        write_rows(inputs['paths'][kind],rows)
    with pytest.raises(ValueError,match='final output hash'): run(inputs)


def test_earlier_malformed_call_cannot_be_hidden_by_a_valid_final_call(tmp_path):
    inputs=fixture(tmp_path); make_failure(inputs,'bad','missing_action_json')
    calls=read_rows(inputs['paths']['calls']); valid=copy.deepcopy(calls[1]); valid['task_id']=calls[0]['task_id']; calls.insert(1,valid)
    write_rows(inputs['paths']['calls'],calls)
    for kind in ('records','outcomes'):
        rows=read_rows(inputs['paths'][kind]); m=rows[0]['metadata']; m[f'hf_torch.{MODEL}.calls']=2
        m[f'hf_torch.{MODEL}.prompt_tokens']=6; m[f'hf_torch.{MODEL}.completion_tokens']+=valid['output_tokens']
        write_rows(inputs['paths'][kind],rows)
    with pytest.raises(ValueError,match='only the final typed'): run(inputs)


def test_failure_adjustment_does_not_overwrite_raw_official_success(tmp_path):
    inputs=fixture(tmp_path,'full_panel'); make_failure(inputs,'bad','missing_action_json')
    for kind in ('records','outcomes'):
        rows=read_rows(inputs['paths'][kind]); card=rows[0]['metadata']['final_scorecard'][0]
        card.update(score=2,scoreNormalized=1.0,completed=True,completedSuccessfully=True); rows[0]['predicted']=1.0
        write_rows(inputs['paths'][kind],rows)
    result=run(inputs)
    assert result['official_completed_successfully_rate']['mean']==pytest.approx(1/120)
    assert result['failure_adjusted_completed_successfully_rate']['mean']==0
    assert result['primary_official_progress_score']['mean']==pytest.approx(60.5/120)


@pytest.mark.parametrize('target', ['task', 'calls', 'records', 'event_abort', 'event_clock', 'source', 'runtime', 'stress'])
def test_missing_provenance_and_infrastructure_failures_rejected(tmp_path,target):
    inputs=fixture(tmp_path)
    if target=='task':
        write_rows(inputs['paths']['outcomes'],read_rows(inputs['paths']['outcomes'])[1:])
    elif target=='calls':
        edit_row(inputs,'calls','messages',[{'role':'user','content':'"scoreNormalized": 0.5'}])
    elif target=='records':
        edit_row(inputs,'records','runtime_contract_sha256','unknown')
    elif target=='event_abort':
        edit_row(inputs,'events','event','infrastructure_abort',index=1)
    elif target=='event_clock':
        edit_row(inputs,'events','monotonic_seconds',0,index=3)
    elif target=='source':
        path=inputs['source']/SOURCE_FILES[0]; path.write_text(path.read_text().replace('self.score / self.maxScore','0.5'))
        inputs['manifest']['endpoint_source_sha256'][SOURCE_FILES[0]]=sha256(path)
    elif target=='runtime': inputs['runtime']['status']='infrastructure_failure'
    elif target=='stress': edit_row(inputs,'stress','output_tokens',399)
    with pytest.raises(ValueError): run(inputs)


def test_task_failure_does_not_excuse_failed_slurm_allocation(tmp_path):
    inputs=fixture(tmp_path); make_failure(inputs,'bad','missing_action_json')
    inputs['resources'].update(state='FAILED',exit_code='1:0')
    with pytest.raises(ValueError,match='infrastructure Slurm exit'): run(inputs)


def test_unseen_runtime_tasks_cannot_overlap_production_seeds(tmp_path):
    inputs=fixture(tmp_path); inputs['manifest']['task_instances'][0]['random_seed']=4
    with pytest.raises(ValueError,match='overlaps production'): run(inputs)


@pytest.mark.parametrize('seeds', [[0], [0,1]])
def test_prospectively_frozen_balanced_timing_fallback_retains_all_strata(tmp_path,seeds):
    inputs=fixture(tmp_path,'full_panel'); manifest=inputs['manifest']
    manifest.update(audit_scope='balanced_census',seeds=seeds,timing_based_fallback=True,
                    selection_without_outcome_rates=True)
    manifest['task_instances']=[row for row in manifest['task_instances'] if row['random_seed'] in seeds]
    ids={f"{row['scenario_name']}_{row['difficulty']}_{row['random_seed']}" for row in manifest['task_instances']}
    manifest['workers'][0]['task_ids']=[task for task in manifest['workers'][0]['task_ids'] if task in ids]
    inputs['runtime']['expected_task_ids']=manifest['workers'][0]['task_ids']
    for kind in ('outcomes','calls','records','events'):
        write_rows(inputs['paths'][kind],[row for row in read_rows(inputs['paths'][kind]) if row['task_id'] in ids])
    result=run(inputs)
    assert result['n_tasks']==24*len(seeds) and result['independent_strata_count']==24
    assert result['official_full_five_seed_panel'] is False
    assert result['primary_official_progress_score']['mean']==0.5
    assert 'timing-only fallback' in result['balanced_census_boundary']
    manifest['selection_without_outcome_rates']=False
    with pytest.raises(ValueError,match='timing-only fallback'): run(inputs)
