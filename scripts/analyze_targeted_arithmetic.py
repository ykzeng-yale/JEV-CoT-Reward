"""Problem-clustered, paired development analysis after the independent record audit."""
import argparse
import json
from pathlib import Path
import random
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def analyze(run, audit_path):
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'passed_targeted_arithmetic_audit':
        raise ValueError('Independent audit must pass first')
    schedule = json.loads((run / 'schedule.json').read_text())
    rows = {(r['problem_id'], r['repeat']): r for r in read_jsonl(run / 'outcomes.jsonl')}
    if len(schedule) != 25 or len(rows) != 50:
        raise ValueError('Incomplete data')
    differences = []
    by_source = {}
    counts = {'correction_success': 0, 'sham_success': 0, 'correction_only': 0, 'sham_only': 0}
    costs = {'generated_tokens': 0, 'processed_prompt_tokens': 0, 'model_service_seconds': 0.0,
             'inserted_instruction_tokens': 0}
    sham_costs = {'generated_tokens': 0, 'processed_prompt_tokens': 0, 'model_service_seconds': 0.0,
                  'inserted_instruction_tokens': 0}
    by_family = {}
    for item in schedule:
        pid = item['problem_id']
        source_rows = {(r['problem_id'], r['repeat']): r for r in
                       read_jsonl(Path(item['source_run']) / 'outcomes.jsonl') if r['action'] == 'sham'}
        ds = []
        for arm in item['arms']:
            row = rows[pid, arm['repeat']]
            y = int(row['outcome']['success'])
            z = int(arm['sham_success'])
            counts['correction_success'] += y
            counts['sham_success'] += z
            counts['correction_only'] += int(y == 1 and z == 0)
            counts['sham_only'] += int(y == 0 and z == 1)
            ds.append(y - z)
            costs['generated_tokens'] += row['generated_tokens']
            costs['processed_prompt_tokens'] += row['prompt_tokens_processed']
            costs['model_service_seconds'] += row['elapsed_seconds']
            costs['inserted_instruction_tokens'] += row['inserted_tokens']
            source = source_rows[pid, arm['repeat']]
            sham_costs['generated_tokens'] += source['generated_tokens']
            sham_costs['processed_prompt_tokens'] += source['prompt_tokens_processed']
            sham_costs['model_service_seconds'] += source['elapsed_seconds']
            sham_costs['inserted_instruction_tokens'] += sum(v.get('tokens', 0) for v in source['overhead'])
        d = statistics.mean(ds)
        differences.append(d)
        by_source.setdefault(item['source_run'], []).append(d)
        by_family.setdefault(item['task']['family'], []).append(d)
    rng = random.Random(20260928)
    samples = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(20000)]
    samples.sort()
    return {
        'status': 'audited_exploratory_development_analysis',
        'problems': 25, 'paired_episodes': 50,
        'counts': counts,
        'correction_minus_prior_sham': {
            'problem_weighted_mean': statistics.mean(differences),
            'descriptive_cluster_bootstrap_95': [samples[499], samples[19499]],
            'by_source_mean': {k: statistics.mean(v) for k, v in by_source.items()},
            'by_source_n': {k: len(v) for k, v in by_source.items()},
        },
        'correction_arm_costs': costs,
        'prior_sham_arm_costs': sham_costs,
        'by_family': {k: {'problems': len(v), 'mean_difference': statistics.mean(v)}
                      for k, v in by_family.items()},
        'limitations': [
            'Selected checkpoints have explicit false additions and are not representative of all reasoning states.',
            'The correction arm receives checked arithmetic information absent from the sham arm.',
            'The sham comparator was saved earlier, so arms were not concurrent despite matched prefixes and seeds.',
            'This experiment tests actionable correction, not Jev-based intervention selection or a complete controller.',
            'Bootstrap interval is descriptive for this development study and does not correct selection or multiplicity.',
            'Correction acquisition and source sham costs must be included in deployment comparisons.',
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse overwrite')
    args.output.write_text(json.dumps(analyze(args.run, args.audit), indent=2) + '\n')
