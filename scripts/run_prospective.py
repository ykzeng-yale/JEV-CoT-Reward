#!/usr/bin/env python3
"""Bounded prospective diagnostic using already frozen trusted local policies."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import analyze_screen as screen
from jev_control.features import QUESTIONS, SCHEMA_VERSION
from jev_control.mlx_backend import MLXBackend
from jev_control.prospective import POLICIES, durable_record, execute_selected
from jev_control.tasks import make_task, verify
from local_judge import judge_prompt, parse_probabilities
from run_development import PROMPT_SUFFIX, boundary_now
from run_mechanism import LoggedBackend, checkpoint_from_initial, model_identity, file_sha256, write_json
from run_screen import rollout

CONFIG = ROOT / 'configs/prospective_diagnostic_v1.json'


def load_frozen(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    blob = (directory / 'controllers.pkl').read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest['artifact_sha256']:
        raise ValueError('Frozen controller digest mismatch')
    for package, version in manifest['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Frozen controller package mismatch')
    for name, digest in manifest['source_sha256'].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError('Frozen controller source mismatch')
    # Only a trusted locally created artifact from freeze_controllers.py is allowed.
    controllers = pickle.loads(blob)
    if set(controllers) != set(POLICIES) - {'always_continue'}:
        raise ValueError('Frozen policy inventory mismatch')
    return controllers, manifest


def predict_actions(controllers, checkpoint, local, budget):
    document, numeric = screen.observable_features(checkpoint, budget)
    choices, probabilities = {'always_continue': 'continue'}, {}
    for name, controller in controllers.items():
        features = numeric
        if name.endswith('plus_jev'):
            features = np.r_[numeric, screen.semantic_features(checkpoint.get('jev'))]
        elif name.endswith('plus_local'):
            features = np.r_[numeric, screen.semantic_features(local)]
        probability = controller.predict([document], [features])[0]
        probabilities[name] = probability.tolist()
        choices[name] = screen.ACTIONS[int(probability.argmax())]
    return choices, probabilities


def local_features(backend, checkpoint, index):
    from mlx_lm.sample_utils import make_sampler
    raw = backend.backend
    old = raw._sampler
    backend.context = {'phase': 'local_judge', 'problem_id': checkpoint['problem_id'], 'temperature': 0}
    try:
        raw._sampler = make_sampler(temp=0, top_p=.9)
        prompt = backend.encode_chat([{'role': 'user', 'content': judge_prompt(checkpoint['state'], QUESTIONS)}])
        generation = backend.generate(prompt, 192, 20260927 + index, timeout_s=120)
    finally:
        raw._sampler = old
    record = {'problem_id': checkpoint['problem_id'], 'generation': generation.to_dict(), 'error': None}
    try:
        if generation.finish_reason == 'timeout':
            raise ValueError('timeout')
        record['probabilities'] = parse_probabilities(generation.text)
    except (ValueError, TypeError):
        record.update({'error': 'invalid_local_feature_response', 'probabilities': None})
    return record


def run(args):
    controllers, frozen = load_frozen(args.controllers)
    config = json.loads(CONFIG.read_text())
    if args.output.exists():
        raise FileExistsError('Output exists; no automatic replay')
    started = time.monotonic()
    args.output.mkdir(parents=True)
    source_hashes = {}
    for path in [*sorted((ROOT/'src/jev_control').glob('*.py')), *sorted((ROOT/'scripts').glob('*.py')), CONFIG]:
        relative = path.relative_to(ROOT)
        destination = args.output/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(relative)] = file_sha256(destination)
    shutil.copytree(args.controllers, args.output/'frozen_controllers')
    identity = model_identity(args.model)
    training = json.loads((args.training_run/'manifest.json').read_text())
    if identity['weight_revision_sha256'] != training['model_identity']['weight_revision_sha256']:
        raise ValueError('Receiver differs from training model')
    if file_sha256(args.training_run/'manifest.json') != frozen['training_inputs_sha256']['manifest.json']:
        raise ValueError('Training manifest changed')
    schedule = []
    for index in range(config['problems']):
        task = make_task(index, config['task_seed'])
        task['prompt'] += PROMPT_SUFFIX
        if task['id'] in frozen['training_problem_ids']:
            raise ValueError('Training/test task overlap')
        schedule.append({'index': index, 'task': task, 'initial_seed': config['task_seed']+index,
                         'continuation_seed': config['task_seed']+index*10000})
    write_json(args.output/'schedule.json', schedule)
    write_json(args.output/'rubric.json', {'schema': SCHEMA_VERSION, 'questions': QUESTIONS})
    manifest = {'protocol': config['protocol'], 'config': config, 'status': 'initializing',
                'started_unix': time.time(), 'model_identity': identity, 'source_sha256': source_hashes,
                'controller_artifact_sha256': frozen['artifact_sha256'],
                'schedule_sha256': file_sha256(args.output/'schedule.json'), 'max_walltime_seconds': args.wall_seconds}
    write_json(args.output/'manifest.json', manifest)
    for name in ('generation_started.jsonl','generation_events.jsonl','checkpoints.jsonl','decisions.jsonl','outcomes.jsonl','local_judge.jsonl','predictions.jsonl'):
        (args.output/name).touch()
    backend = None
    completed = 0
    try:
        raw = MLXBackend(args.model, quantization_bits=4)
        manifest.update({'status':'running', 'actual_quantization_config': raw.quantization_config})
        write_json(args.output/'manifest.json', manifest)
        backend = LoggedBackend(raw, args.output, started+args.wall_seconds)
        from jev_control.jev import JevClient
        client = JevClient(stage_cap_usd=1.)
        for item in schedule:
            task, index = item['task'], item['index']
            backend.context = {'phase':'initial','problem_id':task['id']}
            prompt = backend.encode_chat([{'role':'user','content':task['prompt']}])
            initial = backend.generate(prompt, 384, item['initial_seed'], stop_when=lambda ids: boundary_now(backend, ids, 256))
            checkpoint, reason = checkpoint_from_initial(backend, task, prompt, initial, 256, 384)
            acquisition = {p: {'usd':0., 'generated_tokens':0, 'service_seconds':0.} for p in POLICIES}
            local = None
            if checkpoint is not None:
                # Preserve state even if acquisition fails or the process is killed.
                durable_record(args.output/'checkpoint_attempts.jsonl', checkpoint)
                checkpoint['jev'] = client.evaluate(checkpoint['state'], QUESTIONS)
                local = local_features(backend, checkpoint, index)
                durable_record(args.output/'local_judge.jsonl', local)
                decisions, predictions = predict_actions(controllers, checkpoint, local, 1024)
                acquisition['cheap_tfidf_plus_jev'] = {'usd':checkpoint['jev'].get('input_cost_usd'),
                    'service_seconds':checkpoint['jev'].get('elapsed_seconds'), 'generated_tokens':0}
                g = local['generation']
                acquisition['cheap_tfidf_plus_local'] = {'usd':0., 'generated_tokens':g['generated_tokens'],
                    'prompt_tokens':g['prompt_tokens'],'service_seconds':g['elapsed_seconds']}
            else:
                decisions, predictions = {p:'continue' for p in POLICIES}, {}
                checkpoint = {'problem_id':task['id'],'task':task,'prompt_ids':prompt,'retained_ids':initial.token_ids,
                              'initial':initial.to_dict(),'sha256':hashlib.sha256(json.dumps([prompt,initial.token_ids]).encode()).hexdigest()}
            checkpoint['eligibility_reason'] = reason
            durable_record(args.output/'checkpoints.jsonl', checkpoint)
            durable_record(args.output/'predictions.jsonl', {'problem_id':task['id'],'probabilities':predictions})
            def action_run(action, seed):
                backend.context = {'phase':'continuation','problem_id':task['id'],'action':action}
                if initial.finish_reason in ('stop','timeout'):
                    return {'action':action,'seed':seed,'outcome':verify(task,initial.text),'text':initial.text,
                            'generated_tokens':0,'prompt_tokens_processed':0,'elapsed_seconds':0.,'calls':[],'overhead':[]}
                return rollout(backend, task, prompt, initial.token_ids, action, seed, 1024-initial.generated_tokens, 96)
            execute_selected(task['id'], decisions, item['continuation_seed'], initial.to_dict(),
                             args.output/'decisions.jsonl', args.output/'outcomes.jsonl', action_run,
                             checkpoint_sha256=checkpoint['sha256'], acquisition_costs=acquisition)
            completed += 1
            print(json.dumps({'completed_problems':completed}), flush=True)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update({'status':'failed','error_type':type(exc).__name__})
        raise
    finally:
        manifest.update({'completed_problems':completed,'wall_elapsed_seconds':time.monotonic()-started})
        write_json(args.output/'manifest.json', manifest)
        write_json(args.output/'summary.json', {'status':manifest['status'],'completed_problems':completed,
                   'planned_problems':24,'wall_elapsed_seconds':manifest['wall_elapsed_seconds']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--controllers', required=True, type=Path)
    p.add_argument('--training-run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--wall-seconds', type=float, default=3600)
    a=p.parse_args()
    if not math.isfinite(a.wall_seconds) or a.wall_seconds <= 0:
        p.error('Finite positive deadline required')
    run(a)
