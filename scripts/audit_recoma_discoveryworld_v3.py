#!/usr/bin/env python3
"""Strict terminal audit of durable ReCoMA-DW v3 records; no inference/API calls.

Runtime qualification never reports production efficacy. Full-panel summaries
retain all 120 assigned tasks, including declared final model-format failures.
Official progress is recomputed from raw score/maxScore; success is read from
source-pinned official scorecards, not inferred from score or SUBMIT text. This
is a record/source audit, not a second simulator/world-state outcome oracle.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics

import numpy as np

SOURCE_FILES = (
    'discoveryworld/TaskScorer.py', 'discoveryworld/DiscoveryWorldAPI.py',
    'discoveryworld/UserInterface.py', 'discoveryworld/ScenarioMaker.py',
    'agents/recoma/react_controller.py',
)
SERIALIZATION_SOURCE_FILES = (
    'recoma/recoma/models/core/generator.py',
    'recoma/recoma/models/impl/hf_torch_generator.py',
)
FAILURE_CODES = frozenset({'missing_action_json', 'invalid_action_json',
                          'non_object_action_json', 'invalid_submit_arguments'})
FORBIDDEN = re.compile(r'"(?:scoreNormalized|scoreCard|criticalHypotheses|'
                       r'criticalQuestions|associatedNotes|completedSuccessfully)"\s*:')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def finite_number(value, name, *, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value)
            and value >= minimum, f'invalid {name}')
    return float(value)


def integer(value, name, *, minimum=0):
    require(type(value) is int and value >= minimum, f'invalid {name}')
    return value


def identity(row):
    scenario = row.get('scenario_name', row.get('scenario'))
    difficulty = row.get('difficulty')
    seed = row.get('random_seed', row.get('seed'))
    require(isinstance(scenario, str) and scenario and isinstance(difficulty, str),
            'task scenario/difficulty identity missing')
    integer(seed, 'task seed')
    return f'{scenario}_{difficulty}_{seed}'


def read_rows(paths, kind):
    rows, hashes = [], {}
    require(bool(paths), f'{kind} files required')
    for path in paths:
        path = Path(path)
        hashes[str(path)] = sha256(path)
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f'{kind} row is not an object')
                rows.append(row)
    return rows, hashes


def unique_rows(rows, kind):
    result = {}
    for row in rows:
        task = identity(row)
        require(row.get('task_id', task) == task, f'{kind} task_id mismatch')
        require(task not in result, f'duplicate {kind} task identity')
        result[task] = row
    return result


def source_contract(manifest, source_root):
    """Check frozen official normalization/scorecard chain without importing it."""
    pins = manifest.get('endpoint_source_sha256', {})
    require(set(pins) == set(SOURCE_FILES), 'all five endpoint/controller source pins required')
    trees, hashes = {}, {}
    for relative in SOURCE_FILES:
        path = Path(source_root) / relative
        require(path.is_file(), f'missing pinned source {relative}')
        hashes[relative] = sha256(path)
        require(hashes[relative] == pins[relative], f'source hash mismatch: {relative}')
        trees[relative] = ast.parse(path.read_text())
    task_class = next((n for n in trees[SOURCE_FILES[0]].body
                       if isinstance(n, ast.ClassDef) and n.name == 'Task'), None)
    require(task_class is not None, 'official Task source missing')
    methods = {n.name: n for n in task_class.body if isinstance(n, ast.FunctionDef)}
    expected = ast.dump(ast.parse('self.score / self.maxScore', mode='eval').body)
    returns = [n.value for n in ast.walk(methods['getScoreNormalized']) if isinstance(n, ast.Return)]
    require(len(returns) == 1 and ast.dump(returns[0]) == expected,
            'official normalized-score formula changed')
    assignments = {}
    for node in ast.walk(methods['taskProgressDict']):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                assignments[target.slice.value] = ast.dump(node.value)
    for key, expression in {'score': 'self.score', 'maxScore': 'self.maxScore',
                            'scoreNormalized': 'self.getScoreNormalized()',
                            'completed': 'self.completed',
                            'completedSuccessfully': 'self.completedSuccessfully'}.items():
        require(assignments.get(key) == ast.dump(ast.parse(expression, mode='eval').body),
                f'official scorecard field contract changed: {key}')
    ui_calls = [n for n in ast.walk(trees[SOURCE_FILES[2]]) if isinstance(n, ast.FunctionDef)
                and n.name == 'getFullTaskProgressJSON']
    require(len(ui_calls) == 1 and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'taskProgressDict' for n in ast.walk(ui_calls[0])),
            'official UI scorecard chain changed')
    api_methods = [n for n in ast.walk(trees[SOURCE_FILES[1]]) if isinstance(n, ast.FunctionDef)
                   and n.name == 'getTaskScorecard']
    require(len(api_methods) == 1 and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'getFullTaskProgressJSON' for n in ast.walk(api_methods[0])),
            'official API scorecard chain changed')
    for node in trees[SOURCE_FILES[3]].body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCENARIO_INFOS'
                                               for t in node.targets):
            scenarios = ast.literal_eval(node.value)
            break
    else:
        raise ValueError('official scenario inventory missing')
    parser = trees[SOURCE_FILES[4]]
    values = {n.targets[0].attr: n.value.value for n in ast.walk(parser)
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Attribute) and isinstance(n.value, ast.Constant)}
    require(values.get('full_code_regex') == r'```json\n(.*?)```'
            and values.get('partial_code_regex') == r'.*```json\n(.*)',
            'frozen controller JSON extraction contract changed')
    contract_path = Path(source_root).parent / 'recoma/recoma/utils/task_accounting.py'
    require(manifest.get('runtime_contract') == 'recoma_scientific_failure_accounting_v3',
            'frozen scientific-failure runtime contract required')
    require(contract_path.is_file() and sha256(contract_path) == manifest.get('runtime_contract_sha256'),
            'scientific-failure runtime source hash mismatch')
    hashes['recoma/recoma/utils/task_accounting.py'] = sha256(contract_path)
    # The original bundle inventory is immutable even when an independent
    # auditor is corrected. Bind serialization reconstruction to those actual
    # generator/backend bytes, not to a newly asserted output convention.
    bundle = Path(source_root).parent
    inventory_path = bundle / 'SHA256SUMS'
    require(inventory_path.is_file()
            and sha256(inventory_path) == manifest.get('source_manifest_sha256'),
            'frozen runtime source inventory hash mismatch')
    source_inventory = {}
    for line in inventory_path.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        require(match is not None, 'invalid frozen runtime source inventory')
        digest, relative = match.groups()
        require(relative not in source_inventory, 'duplicate frozen runtime source path')
        source_inventory[relative] = digest.lower()
    serialization_trees = {}
    for relative in SERIALIZATION_SOURCE_FILES:
        path = bundle / relative
        require(path.is_file() and sha256(path) == source_inventory.get(relative),
                f'frozen message serialization source hash mismatch: {relative}')
        hashes[relative] = sha256(path)
        serialization_trees[relative] = ast.parse(path.read_text())
    core = serialization_trees[SERIALIZATION_SOURCE_FILES[0]]
    extract = [node for node in ast.walk(core)
               if isinstance(node, ast.FunctionDef) and node.name == 'extract_role_messages']
    require(len(extract) == 1 and any(isinstance(node, ast.Dict)
            and [key.value if isinstance(key, ast.Constant) else None for key in node.keys] == ['role', 'content']
            for node in ast.walk(extract[0])),
            'core generator role/content insertion-order contract changed')
    backend = serialization_trees[SERIALIZATION_SOURCE_FILES[1]]
    expressions = {target.id: node.value for node in ast.walk(backend)
                   if isinstance(node, ast.Assign) for target in node.targets
                   if isinstance(target, ast.Name) and target.id in ('prompt_text', 'prompt_hash')}
    for name, expression in {
            'prompt_text': 'json.dumps(messages, ensure_ascii=False, separators=(",", ":"))',
            'prompt_hash': 'hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()'}.items():
        require(name in expressions and ast.dump(expressions[name]) == ast.dump(ast.parse(expression, mode='eval').body),
                f'backend prompt serialization/hash contract changed: {name}')
    append = [node for node in backend.body
              if isinstance(node, ast.FunctionDef) and node.name == '_append_jsonl']
    require(len(append) == 1 and any(isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'json' and node.func.attr == 'dumps'
            and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'row'
            and any(keyword.arg == 'sort_keys' and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in node.keywords)
            for node in ast.walk(append[0])),
            'backend durable recursive-key-sort serialization contract changed')
    return scenarios, hashes


def reconstruct_role_messages(messages):
    """Restore the pinned generator's insertion order lost in sorted JSONL.

    Recursive log sorting changes object key order, never message sequence,
    roles or content. Additional fields are not silently dropped or normalized.
    """
    require(isinstance(messages, list) and bool(messages), 'full durable prompt missing')
    result = []
    for message in messages:
        require(isinstance(message, dict) and set(message) == {'role', 'content'}
                and isinstance(message['role'], str) and bool(message['role'])
                and isinstance(message['content'], str), 'invalid durable role/content message schema')
        result.append({'role': message['role'], 'content': message['content']})
    return result


def expected_panel(manifest, scenarios):
    scope = manifest.get('audit_scope')
    require(scope in ('full_panel', 'balanced_census', 'runtime_qualification'), 'explicit audit_scope required')
    rows = manifest.get('task_instances')
    require(isinstance(rows, list) and rows, 'explicit frozen task inventory required')
    expected = unique_rows(rows, 'manifest')
    for row in rows:
        scenario = row.get('scenario_name', row.get('scenario'))
        require(scenario in scenarios and row['difficulty'] in scenarios[scenario]['difficulty'],
                'task is not a supported official scenario/difficulty')
    if scope in ('full_panel', 'balanced_census'):
        panel_seeds = manifest.get('seeds')
        if scope == 'full_panel':
            require(panel_seeds == [0, 1, 2, 3, 4] and len(rows) == 120,
                    'full panel must retain 120 tasks and official seeds 0-4')
        else:
            require(panel_seeds in ([0], [0, 1]) and len(rows) == 24 * len(panel_seeds),
                    'balanced fallback must retain all 24 strata at one/two frozen seeds')
            require(manifest.get('timing_based_fallback') is True
                    and manifest.get('selection_without_outcome_rates') is True,
                    'balanced census must be explicitly frozen as a timing-only fallback')
        strata = defaultdict(list)
        for row in rows:
            strata[(row.get('scenario_name', row.get('scenario')), row['difficulty'])].append(
                row.get('random_seed', row.get('seed')))
        require(len(strata) == 24 and all(sorted(v) == panel_seeds for v in strata.values()),
                'task census must cover every frozen-seed stratum exactly once')
        require({k[1] for k in strata} == {'Easy', 'Normal', 'Challenge'}
                and len({k[0] for k in strata}) == 8, 'full panel is not the 8x3 benchmark')
    else:
        require(len(rows) == 2 and manifest.get('seeds') == [5],
                'qualification requires exactly two separately frozen seed-5 tasks')
        require(manifest.get('qualification_only') is True,
                'qualification must explicitly exclude efficacy/production pooling')
        for row in rows:
            scenario = row.get('scenario_name', row.get('scenario'))
            require(row.get('random_seed', row.get('seed')) == 5
                    and '5' in scenarios[scenario]['variations'],
                    'qualification seed is unsupported or overlaps production')
    return scope, expected


def classify_output(text):
    """Independently reproduce frozen extraction and typed-format predicates."""
    require(isinstance(text, str), 'call output text missing')
    controller_text = text.lstrip()
    full = re.search(r'```json\n(.*?)```', controller_text, re.DOTALL)
    if full:
        payload = full.group(1).strip()
    else:
        partial = re.match(r'.*```json\n(.*)', controller_text, re.DOTALL)
        if partial and partial.group(1).strip():
            payload = partial.group(1).strip()
        else:
            try:
                json.loads(controller_text)
                payload = controller_text
            except json.JSONDecodeError:
                return 'missing_action_json', None
    try:
        action = json.loads(payload)
    except json.JSONDecodeError:
        return 'invalid_action_json', None
    if not isinstance(action, dict):
        return 'non_object_action_json', None
    if action.get('action') == 'SUBMIT' and (
            not isinstance(action.get('arg1'), str)
            or not isinstance(action.get('thought', ''), str)):
        return 'invalid_submit_arguments', action
    return None, action


def audit_events(events, records):
    require(len(events) == 2 * len(records), 'durable task event coverage mismatch')
    states, order = {}, []
    for event in events:
        kind, task = event.get('event'), event.get('task_id')
        require(kind in ('task_started', 'task_ended'), 'infrastructure abort or unknown task event')
        finite_number(event.get('monotonic_seconds'), 'event monotonic time')
        require(task in records, 'event task is outside assigned panel')
        if kind == 'task_started':
            require(task not in states, 'task retry/resample or duplicate start')
            states[task] = event
        else:
            require(task in states and states[task]['event'] == 'task_started',
                    'ended task lacks its unique start')
            require(event['monotonic_seconds'] >= states[task]['monotonic_seconds'],
                    'task ended before start')
            require(event.get('scientific_task_status') == records[task]['scientific_task_status'],
                    'task event/record failure status mismatch')
            states[task] = event
            order.append(task)
    require(set(states) == set(records) and all(e['event'] == 'task_ended' for e in states.values()),
            'a task has no durable terminal event')
    require(order == list(records), 'durable record order differs from task-end order')


def audit_event_shards(paths):
    # Monotonic clocks are worker-local; never compare clocks across workers.
    for path in paths:
        events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        require(len(events) % 2 == 0, 'incomplete durable worker event pair')
        previous = -1.0
        for index in range(0, len(events), 2):
            start, end = events[index:index + 2]
            require(start.get('event') == 'task_started' and end.get('event') == 'task_ended'
                    and start.get('task_id') == end.get('task_id'),
                    'worker retry, interleaving or infrastructure abort event')
            for event in (start, end):
                current = finite_number(event.get('monotonic_seconds'), 'worker event time')
                require(current >= previous, 'worker event clock went backwards')
                previous = current


def official_endpoint(row):
    card = row.get('metadata', {}).get('final_scorecard')
    require(isinstance(card, list) and len(card) == 1 and isinstance(card[0], dict),
            'unique official terminal scorecard missing')
    c = card[0]
    score = finite_number(c.get('score'), 'raw official score')
    maximum = finite_number(c.get('maxScore'), 'official maxScore', minimum=1e-300)
    require(score <= maximum, 'official score exceeds maximum')
    normalized = finite_number(c.get('scoreNormalized'), 'official normalized score')
    require(math.isclose(normalized, score / maximum, rel_tol=1e-12, abs_tol=1e-12),
            'normalized score differs from independently recomputed score/maxScore')
    require(type(c.get('completed')) is bool and type(c.get('completedSuccessfully')) is bool,
            'official completion flags must be Boolean')
    require(not c['completedSuccessfully'] or c['completed'],
            'official success without completion')
    require(isinstance(c.get('taskName'), str) and c['taskName'], 'official taskName missing')
    require(isinstance(c.get('scoreCard'), list), 'official component scorecard missing')
    for component in c['scoreCard']:
        require(isinstance(component, dict), 'invalid official scorecard component')
        component_score = finite_number(component.get('score'), 'component score')
        component_max = finite_number(component.get('maxScore'), 'component maximum')
        require(component_score <= component_max and type(component.get('completed')) is bool,
                'invalid component score/completion')
    # The public answerer returns normalized score; do not use its EM summary.
    predicted = row.get('predicted')
    try:
        predicted_float = float(predicted)
    except (TypeError, ValueError):
        raise ValueError('answerer prediction is not its terminal normalized score') from None
    require(type(predicted) in (str, int, float) and math.isfinite(predicted_float)
            and math.isclose(predicted_float, normalized, rel_tol=1e-12, abs_tol=1e-12),
            'answerer prediction differs from official endpoint')
    return normalized, c['completedSuccessfully']


def descriptive_summary(values):
    rng = np.random.default_rng(20260930)
    data = np.asarray(values, dtype=float)
    samples = rng.choice(data, size=(20000, len(data)), replace=True).mean(axis=1)
    return {'mean': float(data.mean()), 'descriptive_stratum_bootstrap_range':
            [float(x) for x in np.quantile(samples, [.025, .975])],
            'population_confidence_interval': False}


def audit_model_content(manifest, reports, allocation_elapsed, *, checksum_path,
                        verifier_path, before_path, after_path):
    """Bind both compute receipts to the actual frozen verifier and inventory.

    This checks the receipts emitted by the independently frozen byte verifier;
    it does not load weights or pretend to authenticate a remote commit tree.
    The two content inventories must agree with every supplied frozen file SHA,
    even if two self-reported receipt summaries agree with one another.
    """
    contract = manifest.get('model_content_contract')
    require(isinstance(contract, dict)
            and contract.get('snapshot_revision') == manifest['model_revision']
            and contract.get('require_compute_allocation_pre_and_post_verification') is True
            and contract.get('verification_latency_is_allocated_cost') is True,
            'production requires the frozen pre/post model-content contract')
    paths = {'checksums': checksum_path, 'verifier': verifier_path,
             'before': before_path, 'after': after_path}
    require(all(value is not None and Path(value).is_file() for value in paths.values()),
            'production requires frozen model artifacts and both model-content receipts')
    hashes = {key: sha256(value) for key, value in paths.items()}
    require(hashes['checksums'] == contract.get('checksums_sha256')
            and hashes['verifier'] == contract.get('verifier_sha256'),
            'production frozen model checksum/verifier artifact hash mismatch')
    expected = {}
    for line in Path(checksum_path).read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        require(match is not None, 'invalid frozen model checksum manifest')
        digest, name = match.groups()
        relative = PurePosixPath(name)
        require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
                and '\\' not in name and '\x00' not in name,
                'unsafe frozen model checksum path')
        name = relative.as_posix()
        require(name not in expected, 'duplicate frozen model checksum path')
        expected[name] = digest.lower()
    require(bool(expected), 'empty frozen model checksum inventory')
    inventories, receipts = [], []
    elapsed = cpu = 0.0
    for phase in ('before', 'after'):
        receipt = json.loads(Path(paths[phase]).read_text())
        require(isinstance(receipt, dict) and receipt.get('status') == 'PASS_MODEL_SNAPSHOT_CONTENT'
                and receipt.get('schema') == 'model_snapshot_content_verification_v3',
                f'{phase} production model-content verification did not pass')
        require(receipt.get('snapshot_revision') == manifest['model_revision']
                and receipt.get('checksum_file_sha256') == hashes['checksums']
                and receipt.get('verifier_sha256') == hashes['verifier'],
                f'{phase} production model-content identity mismatch')
        snapshot = receipt.get('snapshot_path')
        resolved = receipt.get('resolved_snapshot_path')
        require(isinstance(snapshot, str) and Path(snapshot).is_absolute()
                and Path(snapshot).name == manifest['model_revision']
                and isinstance(resolved, str) and Path(resolved).is_absolute()
                and Path(resolved).name == manifest['model_revision'],
                f'{phase} production model snapshot path mismatch')
        require(all(report.get('model_path') == resolved for report in reports),
                f'{phase} model-content receipt does not bind the loaded runtime snapshot')
        require(receipt.get('model_loaded') is False and receipt.get('model_imports') is False,
                'content verifier must not load/import a model')
        for counter in ('downloaded_bytes', 'model_generations', 'hosted_calls', 'jev_calls'):
            require(integer(receipt.get(counter), 'content verification ' + counter) == 0,
                    'content verification introduced unaccounted compute/network calls')
        elapsed += finite_number(receipt.get('elapsed_seconds'), 'content verification wall seconds')
        cpu += finite_number(receipt.get('process_cpu_seconds'), 'content verification CPU seconds')
        rows = receipt.get('files')
        require(isinstance(rows, list) and rows
                and integer(receipt.get('file_count'), 'verified file count') == len(rows),
                f'{phase} verified content inventory is incomplete')
        inventory, total_bytes, addressed, unaddressed = {}, 0, 0, 0
        unique_paths, distinct_contents = {}, {}
        for row in rows:
            require(isinstance(row, dict), 'model content row is not an object')
            name = row.get('relative_path')
            require(isinstance(name, str) and name in expected and name not in inventory,
                    f'{phase} missing/extra/duplicate model-content file')
            require(row.get('sha256') == row.get('expected_sha256') == expected[name],
                    f'{phase} model-content file differs from the frozen inventory')
            size = integer(row.get('bytes'), 'verified model file bytes')
            total_bytes += size
            target = row.get('resolved_content_path')
            require(row.get('snapshot_reference_path') == str(Path(snapshot) / name)
                    and isinstance(target, str) and Path(target).is_absolute(),
                    f'{phase} model-content reference path mismatch')
            if target in unique_paths:
                require(unique_paths[target] == (expected[name], size),
                        'conflicting verified content identity/byte counts')
            unique_paths[target] = (expected[name], size)
            distinct_contents[(expected[name], size)] = size
            reference_type = row.get('reference_type')
            blob_type = row.get('content_addressed_blob_type')
            blob_name = row.get('content_addressed_blob_name')
            blob_digest = row.get('content_addressed_blob_digest')
            if reference_type == 'regular_file':
                require(row.get('declared_symlink_target') is None and blob_type is None
                        and blob_name is None and blob_digest is None
                        and row.get('content_addressed_blob_verified') is None,
                        'regular file has a contradictory content-addressed symlink claim')
            else:
                require(reference_type == 'symlink' and isinstance(row.get('declared_symlink_target'), str),
                        'verified model reference type is invalid')
                if blob_type is None:
                    require(blob_name is None and blob_digest is None
                            and row.get('content_addressed_blob_verified') is None,
                            'unaddressed symlink has a contradictory blob identity')
                    unaddressed += 1
                else:
                    require(blob_type in ('hf_lfs_sha256', 'hf_git_blob_sha1')
                            and row.get('content_addressed_blob_verified') is True
                            and Path(target).parent.name == 'blobs' and Path(target).name == blob_name
                            and isinstance(blob_name, str) and blob_digest == blob_name
                            and re.fullmatch(r'[0-9a-f]{64}' if blob_type == 'hf_lfs_sha256'
                                             else r'[0-9a-f]{40}', blob_name) is not None,
                            'HF content-addressed symlink identity mismatch')
                    if blob_type == 'hf_lfs_sha256':
                        require(blob_digest == expected[name], 'HF LFS content hash mismatch')
                    addressed += 1
            inventory[name] = (expected[name], size, target, reference_type,
                               row.get('declared_symlink_target'), blob_type, blob_name, blob_digest)
        require(set(inventory) == set(expected), f'{phase} verified content inventory omits frozen model files')
        # The byte verifier deduplicates by (device,inode), which its receipt
        # intentionally does not expose. Different resolved paths can be valid
        # hard links. Check identifiable bounds rather than inventing inode IDs.
        unique_count = integer(receipt.get('unique_content_files'), 'unique content file count', minimum=1)
        unique_bytes = integer(receipt.get('unique_content_bytes'), 'unique content bytes')
        require(integer(receipt.get('total_verified_reference_bytes'), 'total verified reference bytes') == total_bytes
                and len(distinct_contents) <= unique_count <= len(unique_paths)
                and sum(distinct_contents.values()) <= unique_bytes <= sum(x[1] for x in unique_paths.values())
                and integer(receipt.get('hf_content_addressed_symlinks_verified'), 'verified HF symlink count') == addressed
                and integer(receipt.get('non_content_addressed_symlinks'), 'unaddressed symlink count') == unaddressed,
                'model-content receipt aggregate counters disagree with its full file inventory')
        inventories.append(inventory)
        receipts.append(receipt)
    require(inventories[0] == inventories[1]
            and receipts[0]['snapshot_path'] == receipts[1]['snapshot_path']
            and receipts[0]['resolved_snapshot_path'] == receipts[1]['resolved_snapshot_path']
            and receipts[0]['unique_content_files'] == receipts[1]['unique_content_files']
            and receipts[0]['unique_content_bytes'] == receipts[1]['unique_content_bytes'],
            'pre/post model content or references changed during the production allocation')
    require(elapsed <= allocation_elapsed + 1.0,
            'model verification time exceeds its allocated Slurm wall time')
    return {'status': 'PASS_PRE_POST_FROZEN_MODEL_CONTENT', 'input_sha256': hashes,
            'input_paths': {key: str(value) for key, value in paths.items()},
            'file_count': len(expected), 'total_verified_reference_bytes': receipts[0]['total_verified_reference_bytes'],
            'snapshot_revision': manifest['model_revision'], 'verification_elapsed_seconds': elapsed,
            'verification_process_cpu_seconds': cpu, 'latency_included_in_allocation_cost': True,
            'claim_boundary': 'Frozen local file bytes and applicable HF blob identities verified before/after model work; no remote commit-tree attestation, inference-quality or scientific efficacy claim.'}


def audit(manifest, outcome_paths, call_paths, resources, tokenizer_path, *,
          task_record_paths, event_paths, source_root, runtime_report=None, stress_paths=None, tokenizer=None,
          model_checksum_path=None, model_verifier_path=None,
          model_verification_before=None, model_verification_after=None):
    scenarios, source_hashes = source_contract(manifest, source_root)
    scope, expected = expected_panel(manifest, scenarios)
    outcome_rows, outcome_hashes = read_rows(outcome_paths, 'all_data')
    record_rows, record_hashes = read_rows(task_record_paths, 'task_records')
    events, event_hashes = read_rows(event_paths, 'task_events')
    outcomes = unique_rows(outcome_rows, 'all_data')
    records = unique_rows(record_rows, 'durable record')
    require(set(outcomes) == set(records) == set(expected),
            'complete assigned denominator required; missing/extra task records')
    audit_events(events, records)
    audit_event_shards(event_paths)
    calls, call_hashes = read_rows(call_paths, 'calls')
    grouped = defaultdict(list)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                                  trust_remote_code=False, use_fast=False)
    total_prompt = total_output = 0
    service_seconds = 0.0
    action_counts, thought_calls = Counter(), 0
    for call in calls:
        task = call.get('task_id')
        require(task in expected and call.get('status') == 'ok',
                'unexpected task or infrastructure/model-call failure')
        require(call.get('model') == manifest['model'], 'model identity mismatch')
        require(call.get('model_revision') == manifest['model_revision'], 'call model revision mismatch')
        require(call.get('controller_output_sha256') == digest_text(call.get('output_text', '').lstrip()),
                'controller output identity mismatch')
        call_allocated = finite_number(call.get('peak_allocated_gpu_bytes'), 'call peak allocated memory', minimum=1)
        call_reserved = finite_number(call.get('peak_reserved_gpu_bytes'), 'call peak reserved memory', minimum=1)
        require(call_reserved >= call_allocated, 'call reserved memory below allocation')
        require(call.get('max_new_tokens') == manifest['max_new_tokens_per_generation']
                and call.get('temperature') == manifest['temperature'], 'generation contract mismatch')
        messages = reconstruct_role_messages(call.get('messages'))
        for message in messages:
            require(FORBIDDEN.search(message['content']) is None, 'evaluator field exposed to controller')
        canonical = json.dumps(messages, ensure_ascii=False, separators=(',', ':'))
        require(digest_text(canonical) == call.get('prompt_sha256'), 'prompt hash mismatch')
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors='pt')
        prompt = integer(call.get('input_tokens'), 'input token count', minimum=1)
        require(int(tokens.shape[-1]) == prompt, 'prompt token count mismatch')
        require(prompt <= integer(manifest.get('qualified_context_token_ceiling'), 'qualified context ceiling', minimum=1),
                'episode prompt exceeds independently qualified context envelope')
        ids = call.get('output_token_ids')
        require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids),
                'exact completion token IDs missing')
        output = integer(call.get('output_tokens'), 'output token count')
        require(output == len(ids) and output <= manifest['max_new_tokens_per_generation'],
                'completion token count/cap mismatch')
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        for stop in manifest['stop_sequences']:
            if stop in decoded:
                decoded = decoded.split(stop, 1)[0]
        require(decoded == call.get('output_text'), 'decoded output/stop processing mismatch')
        failure_code, action = classify_output(decoded)
        grouped[task].append((call, failure_code))
        if failure_code is None:
            if isinstance(action.get('thought'), str) and action['thought'].strip():
                thought_calls += 1
            action_counts[str(action.get('action', 'dialog_or_other_object'))] += 1
        total_prompt += prompt
        total_output += output
        service_seconds += finite_number(call.get('elapsed_seconds'), 'call elapsed time')
    require(set(grouped) == set(expected), 'durable calls do not cover assigned tasks')
    raw_scores, raw_success, adjusted_success, steps = [], [], [], []
    strata, failure_counts, task_summaries = defaultdict(list), Counter(), []
    prefix = f"hf_torch.{manifest['model']}"
    for task, record in records.items():
        row = outcomes[task]
        metadata = record.get('metadata')
        require(record.get('runtime_contract') == manifest['runtime_contract']
                and record.get('runtime_contract_sha256') == manifest['runtime_contract_sha256'],
                'durable runtime contract mismatch')
        finite_number(record.get('task_wall_seconds'), 'task wall time')
        require(isinstance(metadata, dict) and metadata == row.get('metadata'),
                'durable/terminal metadata mismatch')
        # Upstream dump_predictions preserves scalar JSON as the original
        # prediction string; durable v3 task accounting parses that same scalar.
        # Validate both representations against the source-derived endpoint,
        # rather than comparing a numeric scalar with its string serialization.
        score, success = official_endpoint(record)
        terminal_score, terminal_success = official_endpoint(row)
        require(score == terminal_score and success == terminal_success,
                'durable/terminal official endpoint mismatch')
        task_calls = grouped[task]
        require(0 < len(task_calls) <= manifest['max_llm_calls_per_episode'], 'task call cap exceeded')
        require(integer(metadata.get(prefix + '.calls'), 'terminal call counter') == len(task_calls),
                'terminal call counter mismatch')
        for field, key in (('prompt_tokens', 'input_tokens'), ('completion_tokens', 'output_tokens')):
            require(integer(metadata.get(prefix + '.' + field), 'terminal token counter') ==
                    sum(c[key] for c, _ in task_calls), 'terminal token counter mismatch')
        n_steps = integer(metadata.get('num_steps'), 'terminal environment actions')
        require(n_steps <= manifest['max_environment_actions_per_episode'], 'environment action cap exceeded')
        failure = metadata.get('scientific_task_failure')
        bad = [(i, code) for i, (_, code) in enumerate(task_calls) if code is not None]
        if failure is None:
            require(not bad and record.get('scientific_task_status') == 'completed',
                    'malformed output lacks declared scientific failure')
            require(metadata.get('scientific_task_status', 'completed') == 'completed',
                    'metadata failure status mismatch')
        else:
            require(isinstance(failure, dict) and set(failure) == {'code', 'output_sha256', 'retry_or_resample'},
                    'typed scientific failure schema mismatch')
            require(failure.get('code') in FAILURE_CODES and failure.get('retry_or_resample') is False,
                    'undeclared scientific failure or retry')
            require(bad == [(len(task_calls)-1, failure['code'])],
                    'scientific failure must bind only the final typed malformed call')
            require(failure['output_sha256'] == digest_text(task_calls[-1][0]['output_text'].lstrip()),
                    'scientific failure final output hash mismatch')
            require(record.get('scientific_task_status') == metadata.get('scientific_task_status') == 'scientific_failure',
                    'scientific failure status mismatch')
            failure_counts[failure['code']] += 1
        adjusted = success and failure is None
        require(type(record.get('failure_adjusted_completed_successfully')) is bool
                and record['failure_adjusted_completed_successfully'] == adjusted,
                'failure-adjusted success status mismatch')
        raw_scores.append(score); raw_success.append(int(success)); adjusted_success.append(int(adjusted))
        steps.append(n_steps)
        key = (record['scenario_name'], record['difficulty'])
        strata[key].append((score, int(success), int(adjusted)))
        task_summaries.append({'task_id': task, 'scientific_task_status': record['scientific_task_status'],
                               'official_progress_score': score, 'official_completed_successfully': success,
                               'failure_adjusted_completed_successfully': adjusted,
                               'environment_actions': n_steps, 'model_calls': len(task_calls)})
    require(resources.get('state') == 'COMPLETED' and resources.get('exit_code') == '0:0',
            'terminal infrastructure Slurm exit must be COMPLETED 0:0')
    request = manifest['resources']
    for key, field in (('account','account'), ('gpu_count','gpu_count'), ('cpus','cpu_count'), ('memory_gib','memory_gib')):
        require(resources.get(field) == request[key], f'Slurm {field} differs from frozen request')
    require(resources.get('qos', 'normal') == 'normal' and resources.get('partition') in request['partitions'],
            'Slurm tier/partition differs from normal frozen request')
    require(resources.get('model_revision') == manifest['model_revision'], 'loaded model revision mismatch')
    elapsed = finite_number(resources.get('elapsed_seconds'), 'Slurm elapsed time', minimum=1e-300)
    require(elapsed <= request['wall_minutes'] * 60, 'Slurm allocation exceeds frozen wall cap')
    runtime = runtime_report if runtime_report is not None else resources.get('runtime')
    reports = runtime if isinstance(runtime, list) else [runtime]
    workers = manifest.get('workers')
    require(isinstance(workers, list) and workers, 'frozen worker assignments required')
    assigned = [task for worker in workers for task in worker.get('task_ids', [])]
    require(len(assigned) == len(set(assigned)) and set(assigned) == set(expected),
            'frozen worker assignments do not partition the task denominator')
    require(len(reports) == len(workers), 'one complete runtime report per worker required')
    allocated_peaks, reserved_peaks = [], []
    for index, report in enumerate(reports):
        require(isinstance(report, dict) and report.get('status') == 'completed'
                and report.get('performed_generation') is True, 'worker infrastructure did not complete')
        require(report.get('worker_index', index) == index
                and report.get('expected_task_ids') == workers[index]['task_ids'],
                'worker runtime/task assignment identity mismatch')
        for field, expected_value in (('model_revision', manifest['model_revision']),
                ('torch', manifest['runtime']['torch_version']),
                ('transformers', manifest['runtime']['transformers_version'])):
            actual = report.get(field)
            if field == 'torch' and isinstance(actual, str):
                actual = actual.split('+')[0]
            require(actual == expected_value, f'runtime {field} mismatch')
        require(report.get('cuda_device_count') == 1, 'worker must see exactly one allocated CUDA GPU')
        require(Path(report.get('model_path', '')).name == manifest['model_revision'],
                'runtime snapshot identity mismatch')
        a = finite_number(report.get('max_memory_allocated_bytes'), 'peak allocated GPU memory', minimum=1)
        r = finite_number(report.get('max_memory_reserved_bytes'), 'peak reserved GPU memory', minimum=1)
        require(r >= a, 'reserved GPU memory below allocated memory')
        require(isinstance(report.get('cuda_device'), str) and report['cuda_device'], 'runtime GPU identity missing')
        if scope in ('full_panel', 'balanced_census'):
            qualified = manifest.get('qualified_gpu')
            total = integer(report.get('cuda_total_memory_bytes'), 'measured physical CUDA memory', minimum=1)
            require(isinstance(qualified, dict) and report['cuda_device'] == qualified.get('name')
                    and total == qualified.get('total_memory_bytes'),
                    'production GPU differs from the measured qualified name/capacity')
            require(r <= total, 'production peak GPU memory exceeds physical capacity')
        finite_number(report.get('elapsed_seconds'), 'worker elapsed time', minimum=1e-300)
        worker_calls = [c for c in calls if c['task_id'] in workers[index]['task_ids']]
        require(a >= max(c['peak_allocated_gpu_bytes'] for c in worker_calls)
                and r >= max(c['peak_reserved_gpu_bytes'] for c in worker_calls),
                'runtime peaks exclude episode calls')
        allocated_peaks.append(a); reserved_peaks.append(r)
    require(len(reports) <= request['gpu_count'], 'workers exceed frozen GPU allocation')
    allocated, reserved = max(allocated_peaks), max(reserved_peaks)
    stress_prompt = stress_output = 0
    stress_seconds = 0.0
    stress_hashes = {}
    if scope == 'runtime_qualification':
        require(request['gpu_count'] == 1 and len(reports) == 1, 'qualification must use one GPU/worker')
        require(bool(stress_paths), 'context stress files required')
        stress_rows = [json.loads(Path(p).read_text()) for p in stress_paths]
        stress_hashes = {str(p):sha256(p) for p in stress_paths}
        require(len(stress_rows) == 1, 'exactly one frozen context-stress generation required')
        stress = stress_rows[0]
        require(stress.get('status') == 'completed' and stress.get('model') == manifest['model']
                and stress.get('model_revision') == manifest['model_revision'], 'context stress identity/exit mismatch')
        for field in ('input', 'output'):
            ids = stress.get(field + '_token_ids')
            require(isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids),
                    'exact context stress token IDs required')
            require(integer(stress.get(field + '_tokens'), 'context stress token count') == len(ids)
                    == manifest['context_stress_' + field + '_tokens'], 'context stress token count/cap mismatch')
        require(tokenizer.decode(stress['output_token_ids'], skip_special_tokens=True) == stress.get('output_text'),
                'context stress decoded output mismatch')
        require(stress.get('max_new_tokens') == manifest['context_stress_output_tokens']
                and stress.get('do_sample') is False, 'context stress generation configuration mismatch')
        stress_prompt, stress_output = stress['input_tokens'], stress['output_tokens']
        stress_seconds = finite_number(stress.get('elapsed_seconds'), 'context stress elapsed time', minimum=1e-300)
        require(digest_text(json.dumps(stress['input_token_ids'], separators=(',', ':'))) ==
                stress.get('input_token_ids_sha256'), 'context stress input token hash mismatch')
        require(stress.get('scientific_episode') is False and stress.get('model_calls') == 1
                and stress.get('jev_calls') == stress.get('hosted_calls') == 0,
                'context stress must be one local runtime-only call')
        require(digest_text(json.dumps(stress['output_token_ids'], separators=(',', ':'))) ==
                stress.get('output_token_ids_sha256'), 'context stress output token hash mismatch')
        stress_a = finite_number(stress.get('peak_allocated_gpu_bytes'), 'context stress allocated GPU memory', minimum=1)
        stress_r = finite_number(stress.get('peak_reserved_gpu_bytes'), 'context stress reserved GPU memory', minimum=1)
        require(stress_r >= stress_a and allocated >= stress_a and reserved >= stress_r,
                'runtime peaks exclude context stress or stress memory invalid')
        require(reports[0].get('context_stress') == {k: v for k, v in stress.items()
                if k not in ('input_token_ids', 'output_token_ids', 'output_text')},
                'runtime/context stress receipt mismatch')
    else:
        require(not stress_paths, 'runtime qualification stress cannot be pooled into production outcomes')
    model_content = None
    if scope in ('full_panel', 'balanced_census'):
        model_content = audit_model_content(manifest, reports, elapsed,
            checksum_path=model_checksum_path, verifier_path=model_verifier_path,
            before_path=model_verification_before, after_path=model_verification_after)
    all_prompt, all_output = total_prompt + stress_prompt, total_output + stress_output
    all_service = service_seconds + stress_seconds
    result = {'audit_status':'PASS', 'audit_scope':scope, 'n_tasks':len(expected),
        'scientific_failure_count':sum(failure_counts.values()), 'scientific_failure_codes':dict(failure_counts),
        'model_calls':len(calls), 'prompt_tokens':total_prompt, 'completion_tokens':total_output,
        'context_stress_prompt_tokens':stress_prompt, 'context_stress_completion_tokens':stress_output,
        'all_prompt_tokens_including_runtime_stress':all_prompt,
        'all_generated_tokens_including_failed_final_calls_and_runtime_stress':all_output,
        'model_service_seconds':all_service, 'episode_model_service_seconds':service_seconds,
        'context_stress_model_service_seconds':stress_seconds,
        'emitted_tokens_per_model_service_second':all_output/all_service if all_service else None,
        'emitted_tokens_per_allocation_second':all_output/elapsed,
        'peak_allocated_gpu_bytes':allocated, 'peak_reserved_gpu_bytes':reserved,
        'actual_gpu_hours_from_slurm':elapsed*request['gpu_count']/3600,
        'actual_reserved_cpu_hours_from_slurm':elapsed*request['cpus']/3600,
        'mean_environment_actions':statistics.mean(steps), 'trace_format_diagnostics':
            {'thought_bearing_calls':thought_calls,'action_type_counts':dict(action_counts)},
        'claim_boundary':manifest.get('claim_boundary'),
        'outcome_audit_boundary':'Source-pinned score/max normalization and official scorecard recomputation from durable records; no independent world-state replay or second success oracle.',
        'runtime_source_inventory':{'path':str(Path(source_root).parent/'SHA256SUMS'),
                                    'sha256':manifest['source_manifest_sha256']},
        'prompt_serialization_contract':{'reconstructed_message_key_order':['role','content'],
            'exact_message_key_schema':['role','content'], 'durable_nested_key_sort':True,
            'source_sha256':{key:source_hashes[key] for key in SERIALIZATION_SOURCE_FILES},
            'boundary':'Source-proven JSON object key-order restoration only; message sequence, roles and content are unchanged.'},
        'source_sha256':source_hashes, 'input_sha256':{'all_data':outcome_hashes,
            'task_records':record_hashes,'task_events':event_hashes,'calls':call_hashes, 'context_stress':stress_hashes},
        'slurm_accounting':resources, 'runtime':runtime}
    if scope in ('full_panel', 'balanced_census'):
        result['model_content_verification'] = model_content
        means = [tuple(statistics.mean(x[i] for x in rows) for i in range(3)) for rows in strata.values()]
        result['primary_official_progress_score'] = descriptive_summary([x[0] for x in means])
        result['official_completed_successfully_rate'] = descriptive_summary([x[1] for x in means])
        result['failure_adjusted_completed_successfully_rate'] = descriptive_summary([x[2] for x in means])
        result['per_task_audited_summary'] = task_summaries
        result['independent_strata_count'] = len(strata)
        result['inference_boundary'] = ('Fixed 24-stratum panel, ' + str(len(manifest['seeds'])) +
            ' frozen parametric seeds each; descriptive stratum resampling only. Failure tasks remain in denominator; raw partial progress is never silently zeroed.')
        result['official_full_five_seed_panel'] = scope == 'full_panel'
        if scope == 'balanced_census':
            result['balanced_census_boundary'] = 'Prospectively frozen timing-only fallback over every theme/difficulty stratum; no outcome-dependent selection and not the official full five-seed benchmark panel.'
    else:
        result['qualification_task_audit'] = task_summaries
        result['qualification_boundary'] = 'Two runtime-only seed-5 tasks; no efficacy rate/CI, no production outcome pooling, no multi-GPU scaling qualification.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest','resources','tokenizer-path','source-root','output'):
        parser.add_argument('--'+field, type=Path, required=True)
    for field in ('output-files','call-files','task-record-files','task-event-files'):
        parser.add_argument('--'+field, nargs='+', type=Path, required=True)
    parser.add_argument('--runtime-report', type=Path, nargs='+')
    parser.add_argument('--stress-files', type=Path, nargs='+')
    for field in ('model-checksums', 'model-verifier', 'model-verification-before', 'model-verification-after'):
        parser.add_argument('--'+field, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'refuse to overwrite a prior independent audit')
    result = audit(json.loads(args.manifest.read_text()),args.output_files,args.call_files,
        json.loads(args.resources.read_text()),args.tokenizer_path,
        task_record_paths=args.task_record_files,event_paths=args.task_event_files,source_root=args.source_root,
        runtime_report=[json.loads(p.read_text()) for p in args.runtime_report] if args.runtime_report else None,
        stress_paths=args.stress_files, model_checksum_path=args.model_checksums,
        model_verifier_path=args.model_verifier, model_verification_before=args.model_verification_before,
        model_verification_after=args.model_verification_after)
    result['manifest_sha256'] = sha256(args.manifest)
    result['auditor_sha256'] = sha256(Path(__file__))
    if args.runtime_report:
        result['runtime_report_sha256'] = {str(p):sha256(p) for p in args.runtime_report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('audit_status','audit_scope','n_tasks','scientific_failure_count',
                                         'model_calls','actual_gpu_hours_from_slurm')},sort_keys=True))


if __name__ == '__main__':
    main()
