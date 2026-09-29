"""Independent record and claim-parser audit for arithmetic opportunity v1."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import run_arithmetic_opportunity_screen as runner
from jev_control.tasks import make_task
from analyze_development import LocalTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z * ((p*(1-p)/n + z*z/(4*n*n))**.5) / den
    return [max(0., mid-half), min(1., mid+half)]


def audit(run, tokenizer_path):
    manifest = json.loads((run / 'manifest.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    schedule = json.loads((run / 'schedule.json').read_text())
    require = lambda cond, msg: (_ for _ in ()).throw(ValueError(msg)) if not cond else None
    require(manifest['status'] == summary['status'] == 'complete', 'Study not complete')
    require(schedule == runner.plan() and len(schedule) == 120, 'Frozen schedule mismatch')
    require(manifest['config_sha256'] == sha(ROOT / 'configs/arithmetic_opportunity_v1.json'), 'Config changed')
    require(manifest['protocol_sha256'] == sha(ROOT / 'docs/arithmetic_opportunity_v1_protocol.md'), 'Protocol changed')
    require(manifest['schedule_sha256'] == sha(run / 'schedule.json'), 'Schedule hash mismatch')
    for name, digest in manifest['source_sha256'].items():
        path = run / 'source' / name
        require(path.is_file() and sha(path) == digest, f'Frozen source mismatch: {name}')
    cps = [json.loads(x) for x in (run / 'checkpoints.jsonl').read_text().splitlines() if x]
    skipped = [json.loads(x) for x in (run / 'skipped.jsonl').read_text().splitlines() if x]
    events = [json.loads(x) for x in (run / 'generation_events.jsonl').read_text().splitlines() if x]
    starts = [json.loads(x) for x in (run / 'generation_started.jsonl').read_text().splitlines() if x]
    require(len(cps) + len(skipped) == 120, 'Enrollment count mismatch')
    require(len(events) == len(starts) == 120 and all(e['status'] == 'complete' for e in events), 'Generation call ledger incomplete')
    cp_by_id = {c['problem_id']: c for c in cps}
    skip_by_id = {c['task']['id']: c for c in skipped}
    require(len(cp_by_id) == len(cps) and len(skip_by_id) == len(skipped), 'Duplicate task record')
    tokenizer = LocalTokenizer(tokenizer_path)
    correct = incorrect = absent = 0
    for ordinal, item in enumerate(schedule):
        task = make_task(item['index'], 81277919)
        event, start = events[ordinal], starts[ordinal]
        require(event['phase'] == start['phase'] == 'initial_prefix_only' and
                event['problem_id'] == start['problem_id'] == task['id'] and
                event['index'] == start['index'] == item['index'], 'Generation chronology/task mismatch')
        require(event['prefix_ids'] == start['prefix_ids'] and
                event['max_tokens'] == start['max_tokens'] == 384 and
                event['seed'] == start['seed'] == item['initial_seed'], 'Generation request mismatch')
        gen = event['generation']
        require(gen['seed'] == item['initial_seed'] and gen['generated_tokens'] == len(gen['token_ids']),
                'Generation result/cost mismatch')
        if task['id'] in cp_by_id:
            cp = cp_by_id[task['id']]
            require(cp['task'] == {**task, 'prompt': task['prompt'] + runner.PROMPT_SUFFIX}, 'Task data mismatch')
            require(cp['prompt_ids'] == event['prefix_ids'] and cp['initial'] == gen, 'Checkpoint does not match generated prefix')
            text = tokenizer.decode(cp['retained_ids'])
            # The protocol uses a single newline as a semantic checkpoint
            # boundary; rstrip() below is used only to reconstruct the segment.
            require(text.endswith('\n') and 'FINAL:' not in text,
                    'Checkpoint does not match frozen newline stop rule')
            segment = cp['state']['latest_segment']
            expected = text.rstrip().rsplit('\n\n', 1)[-1]
            require(segment == expected, 'Claim segment differs from exact retained token prefix')
            require(cp['latest_segment_sha256'] == hashlib.sha256(segment.encode()).hexdigest(), 'Segment hash mismatch')
            match = runner.ADDITION.search(segment)
            expected_claim = None
            if match:
                a, b, c = map(int, match.groups())
                expected_claim = {'claim': [a, b, c], 'correct': a+b == c,
                                  'character_span': [match.start(), match.end()]}
            require(cp['addition_candidate'] == expected_claim, 'Addition parser mismatch')
            if expected_claim is None:
                absent += 1
            elif expected_claim['correct']:
                correct += 1
            else:
                incorrect += 1
        else:
            require(task['id'] in skip_by_id, 'Task missing from enrollment')
            require(skip_by_id[task['id']]['initial'] == gen, 'Skipped initial generation mismatch')
    require(summary['eligible'] == len(cps) and summary['ineligible'] == len(skipped), 'Summary enrollment mismatch')
    require(summary['correct_explicit_additions'] == correct and summary['incorrect_explicit_additions'] == incorrect,
            'Summary claim counts mismatch')
    require(summary['with_explicit_addition'] == correct + incorrect, 'Summary statement count mismatch')
    return {'status': 'passed_arithmetic_opportunity_audit', 'attempted': 120,
            'eligible': len(cps), 'ineligible': len(skipped),
            'explicit_addition': correct + incorrect, 'correct': correct, 'incorrect': incorrect,
            'no_explicit_addition': absent, 'incorrect_rate_among_explicit': incorrect/(correct+incorrect) if correct+incorrect else None,
            'incorrect_rate_wilson95_among_explicit': wilson(incorrect, correct+incorrect),
            'incorrect_rate_among_eligible': incorrect/len(cps) if cps else None,
            'incorrect_rate_wilson95_among_eligible': wilson(incorrect, len(cps)),
            'incorrect_count_over_all_attempts': incorrect,
            'event_rate_bounds_over_all_attempts': [incorrect/120, (incorrect+len(skipped))/120],
            'generation_calls': len(events), 'outcome_evaluations': 0, 'jev_calls': 0,
            'input_sha256': {p.name: sha(p) for p in run.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl')},
            'limitations': ['Conditional on one pinned generator, prompt, task family and checkpoint rule.',
                            'Parser only detects the first explicit integer addition equality in the latest segment.',
                            'This is event availability, not judge accuracy or intervention efficacy.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refuse overwrite')
    result = audit(a.run, a.tokenizer)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
