"""Task-only feasibility screen on fresh eight-operand expression tasks."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from jev_control.horizon_tasks import make_horizon_task
from jev_control.mlx_backend import MLXBackend
from run_mechanism import LoggedBackend, checkpoint_from_initial, model_identity, write_json, append_jsonl
from run_development import PROMPT_SUFFIX, boundary_now

CONFIG = ROOT / 'configs/arithmetic_horizon_opportunity_v1.json'
PROTOCOL = ROOT / 'docs/arithmetic_horizon_opportunity_v1_protocol.md'
ADDITION = re.compile(r'(?<!\d)(\d{1,3})\s*\+\s*(\d{1,3})\s*=\s*(\d{1,3})(?!\d)')


def plan():
    return [{'index': i, 'task_id': f'arithmetic_construction_h8-93027119-{i}',
             'initial_seed': 93027119 + i} for i in range(120)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(model, output, wall_seconds):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = [Path(__file__), CONFIG, PROTOCOL, ROOT / 'scripts/run_mechanism.py',
               ROOT / 'scripts/run_development.py', ROOT / 'src/jev_control/horizon_tasks.py',
               ROOT / 'src/jev_control/tasks.py', ROOT / 'src/jev_control/mlx_backend.py']
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(ROOT)
        dest = output / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        source_hashes[str(relative)] = sha(dest)
    schedule = plan()
    write_json(output / 'schedule.json', schedule)
    manifest = {'status': 'initializing', 'config_sha256': sha(CONFIG), 'protocol_sha256': sha(PROTOCOL),
                'schedule_sha256': sha(output / 'schedule.json'), 'source_sha256': source_hashes,
                'task_seed': 93027119, 'attempted_tasks': 120, 'operand_count': 8,
                'temperature': .7, 'top_p': .9, 'checkpoint_target': 256, 'checkpoint_cap': 384,
                'quantization_bits': 4,
                'package_versions': {p: importlib.metadata.version(p) for p in ('mlx', 'mlx-lm', 'transformers')},
                'started_unix': time.time()}
    write_json(output / 'manifest.json', manifest)
    for name in ('checkpoints', 'skipped', 'generation_started', 'generation_events'):
        (output / f'{name}.jsonl').touch()
    started = time.monotonic()
    status = 'failed'
    try:
        manifest['model_identity'] = model_identity(model)
        raw = MLXBackend(model, temperature=.7, top_p=.9, quantization_bits=4)
        manifest['actual_quantization_config'] = raw.quantization_config
        backend = LoggedBackend(raw, output, started + wall_seconds)
        manifest['status'] = 'running'
        write_json(output / 'manifest.json', manifest)
        for item in schedule:
            task = make_horizon_task(item['index'], 93027119)
            if task['id'] != item['task_id']:
                raise ValueError('Task schedule identity mismatch')
            # Witness is held in task.data for independent audit only; the chat
            # template receives task.prompt and cannot see task.data.
            task['prompt'] += PROMPT_SUFFIX
            prompt_ids = backend.encode_chat([{'role': 'user', 'content': task['prompt']}])
            backend.context = {'phase': 'horizon_prefix_only', 'problem_id': task['id'], 'index': item['index']}
            initial = backend.generate(prompt_ids, 384, item['initial_seed'], timeout_s=180,
                                       stop_when=lambda ids: boundary_now(backend, ids, 256))
            checkpoint, reason = checkpoint_from_initial(backend, task, prompt_ids, initial, 256, 384)
            if reason:
                append_jsonl(output / 'skipped.jsonl', {'task': task, 'initial': initial.to_dict(), 'reason': reason})
                continue
            latest = checkpoint['state']['latest_segment']
            match = ADDITION.search(latest)
            candidate = None
            if match:
                a, b, c = map(int, match.groups())
                candidate = {'claim': [a, b, c], 'correct': a + b == c,
                             'character_span': [match.start(), match.end()]}
            full_text = backend.decode(checkpoint['retained_ids'])
            checkpoint.update({'addition_candidate': candidate,
                               'latest_segment_sha256': hashlib.sha256(latest.encode()).hexdigest(),
                               'retained_text_sha256': hashlib.sha256(full_text.encode()).hexdigest()})
            append_jsonl(output / 'checkpoints.jsonl', checkpoint)
            print(json.dumps({'attempted_index': item['index'], 'status': 'eligible',
                              'addition_present': candidate is not None}), flush=True)
        status = 'complete'
    finally:
        write_json(output / 'manifest.json', {**manifest, 'status': status,
                   'finished_unix': time.time(), 'elapsed_seconds': time.monotonic() - started})
        cps = [json.loads(x) for x in (output / 'checkpoints.jsonl').read_text().splitlines() if x]
        skipped = [json.loads(x) for x in (output / 'skipped.jsonl').read_text().splitlines() if x]
        claims = [r['addition_candidate'] for r in cps if r['addition_candidate'] is not None]
        write_json(output / 'summary.json', {
            'status': status, 'attempted': 120, 'eligible': len(cps), 'ineligible': len(skipped),
            'with_explicit_addition': len(claims), 'without_explicit_addition': len(cps) - len(claims),
            'incorrect_explicit_additions': sum(not c['correct'] for c in claims),
            'correct_explicit_additions': sum(c['correct'] for c in claims),
            'advance_gate': 20, 'jev_calls': 0, 'task_outcomes_collected': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wall-seconds', type=float, default=7200)
    a = p.parse_args()
    run(a.model, a.output, a.wall_seconds)
