"""Independent provenance, hidden-witness and equation audit for horizon screen."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import run_arithmetic_horizon_screen as runner
from jev_control.horizon_tasks import make_horizon_task
from jev_control.tasks import verify
from analyze_development import LocalTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, tokenizer_path):
    m = json.loads((run / 'manifest.json').read_text())
    s = json.loads((run / 'summary.json').read_text())
    schedule = json.loads((run / 'schedule.json').read_text())
    def require(test, message):
        if not test:
            raise ValueError(message)
    require(m['status'] == s['status'] == 'complete', 'Run incomplete')
    require(schedule == runner.plan() and len(schedule) == 120, 'Frozen schedule changed')
    require(m['config_sha256'] == sha(ROOT / 'configs/arithmetic_horizon_opportunity_v1.json'), 'Config hash mismatch')
    require(m['protocol_sha256'] == sha(ROOT / 'docs/arithmetic_horizon_opportunity_v1_protocol.md'), 'Protocol hash mismatch')
    require(m['schedule_sha256'] == sha(run / 'schedule.json'), 'Schedule hash mismatch')
    for name, digest in m['source_sha256'].items():
        path = run / 'source' / name
        require(path.is_file() and sha(path) == digest, f'Frozen source mismatch {name}')
    cps = [json.loads(x) for x in (run / 'checkpoints.jsonl').read_text().splitlines() if x]
    skipped = [json.loads(x) for x in (run / 'skipped.jsonl').read_text().splitlines() if x]
    events = [json.loads(x) for x in (run / 'generation_events.jsonl').read_text().splitlines() if x]
    starts = [json.loads(x) for x in (run / 'generation_started.jsonl').read_text().splitlines() if x]
    require(len(cps) + len(skipped) == len(events) == len(starts) == 120, 'Incomplete task/call ledger')
    require(all(e['status'] == 'complete' for e in events), 'A generation call failed')
    cp_map = {r['problem_id']: r for r in cps}
    skip_map = {r['task']['id']: r for r in skipped}
    require(len(cp_map) == len(cps) and len(skip_map) == len(skipped), 'Duplicate task records')
    tok = LocalTokenizer(tokenizer_path)
    correct = incorrect = absent = 0
    for ordinal, item in enumerate(schedule):
        task = make_horizon_task(item['index'], 93027119)
        event, start = events[ordinal], starts[ordinal]
        require(event['phase'] == start['phase'] == 'horizon_prefix_only' and
                event['problem_id'] == start['problem_id'] == task['id'] and
                event['index'] == start['index'] == item['index'], 'Call chronology mismatch')
        require(event['prefix_ids'] == start['prefix_ids'] and event['seed'] == start['seed'] == item['initial_seed'] and
                event['max_tokens'] == start['max_tokens'] == 384, 'Call request mismatch')
        gen = event['generation']
        require(gen['seed'] == item['initial_seed'] and gen['generated_tokens'] == len(gen['token_ids']), 'Call result mismatch')
        require(verify(task, 'FINAL: ' + task['data']['witness'])['success'], 'Independent task witness invalid')
        require(task['data']['witness'] not in task['prompt'], 'Witness leaked into prompt')
        if task['id'] in cp_map:
            cp = cp_map[task['id']]
            expected_task = {**task, 'prompt': task['prompt'] + __import__('run_development').PROMPT_SUFFIX}
            require(cp['task'] == expected_task and cp['initial'] == gen and cp['prompt_ids'] == event['prefix_ids'], 'Checkpoint source mismatch')
            text = tok.decode(cp['retained_ids'])
            require(text.endswith('\n') and 'FINAL:' not in text, 'Checkpoint stop rule mismatch')
            latest = cp['state']['latest_segment']
            require(latest == text.rstrip().rsplit('\n\n', 1)[-1], 'State not reconstructed from retained prefix')
            require(cp['latest_segment_sha256'] == hashlib.sha256(latest.encode()).hexdigest(), 'Segment hash mismatch')
            match = runner.ADDITION.search(latest)
            expected = None
            if match:
                a, b, c = map(int, match.groups())
                expected = {'claim': [a, b, c], 'correct': a+b == c, 'character_span': [match.start(), match.end()]}
            require(cp['addition_candidate'] == expected, 'Equation parser mismatch')
            if expected is None: absent += 1
            elif expected['correct']: correct += 1
            else: incorrect += 1
        else:
            require(task['id'] in skip_map and skip_map[task['id']]['initial'] == gen, 'Missing or mismatched skipped task')
    require(s['eligible'] == len(cps) and s['ineligible'] == len(skipped), 'Summary enrollment mismatch')
    require(s['correct_explicit_additions'] == correct and s['incorrect_explicit_additions'] == incorrect and
            s['with_explicit_addition'] == correct + incorrect, 'Summary count mismatch')
    n = correct + incorrect
    rate = incorrect / n if n else None
    return {'status': 'passed_arithmetic_horizon_audit', 'attempted': 120,
            'eligible': len(cps), 'ineligible': len(skipped), 'with_explicit_addition': n,
            'correct': correct, 'incorrect': incorrect, 'without_explicit_addition': absent,
            'incorrect_rate_among_explicit': rate, 'advance_gate': 20,
            'gate_passed': incorrect >= 20, 'calls': len(events), 'task_outcomes': 0, 'jev_calls': 0,
            'limitations': ['One pinned model, prompt and sampled checkpoint rule; not a population-wide event rate.',
                            'First explicit integer addition only; no adjudication of other reasoning claims.',
                            'Feasibility screen does not test judge accuracy or correction efficacy.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refuse overwrite')
    a.output.write_text(json.dumps(audit(a.run, a.tokenizer), indent=2) + '\n')
