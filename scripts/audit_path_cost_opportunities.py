"""Estimate availability of verifiable path-cost errors in frozen dev checkpoints.

This is a data-availability audit only. It does not inspect final outcomes or
select cases by whether an intervention would help.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [ROOT / 'runs/action-replication-v3-20260928', ROOT / 'runs/sham-replication-v4-20260928']
EQ = re.compile(r'(?<![A-Z])([A-J](?:\s*(?:→|->)\s*[A-J]){2,})\s*[:=]\s*((?:\d+\s*\+\s*)+\d+)\s*=\s*(\d+)')


def inspect():
    record = {'status': 'development_checkpoint_availability_audit', 'runs': [],
              'limitations': ['Regex detects only explicit path-plus-sum equations in latest_segment.',
                              'These counts do not establish that correcting the claim improves task success.',
                              'Development checkpoints are selected by prior experiment eligibility.',
                              'A path may be invalid because of an incorrect edge weight or edge transition.']}
    aggregate = Counter()
    for run in RUNS:
        cp_path = run / 'checkpoints.jsonl'
        digest = hashlib.sha256(cp_path.read_bytes()).hexdigest()
        problems = set(); parsed = set(); wrong = set(); false_arithmetic = set(); graph_inconsistent = set()
        for line in cp_path.read_text().splitlines():
            row = json.loads(line)
            if row['task']['family'] != 'weighted_path':
                continue
            pid = row['problem_id']; problems.add(pid)
            edges = {(u, v): w for u, v, w in row['task']['data']['edges']}
            segment = row['state']['latest_segment']
            for match in EQ.finditer(segment):
                key = (pid, match.group(0))
                if key in parsed:
                    continue
                parsed.add(key)
                nodes = [ord(x) - ord('A') for x in re.findall(r'[A-J]', match.group(1))]
                claimed_sum = int(match.group(3))
                addends = [int(x) for x in re.findall(r'\d+', match.group(2))]
                expression_sum = sum(addends)
                edge_weights = [edges.get((u, v)) for u, v in zip(nodes, nodes[1:])]
                valid_edges = all(w is not None for w in edge_weights)
                if expression_sum != claimed_sum or not valid_edges or claimed_sum != sum(edge_weights):
                    wrong.add(key)
                if expression_sum != claimed_sum:
                    false_arithmetic.add(key)
                if any(w is None for w in edge_weights) or (all(w is not None for w in edge_weights) and addends != edge_weights):
                    graph_inconsistent.add(key)
        row = {'run': run.name, 'checkpoint_sha256': digest, 'weighted_path_problems': len(problems),
               'explicit_path_sum_claims': len(parsed), 'claims_with_checkable_error': len(wrong),
               'arithmetic_sum_errors': len(false_arithmetic), 'graph_or_edge_weight_mismatches': len(graph_inconsistent)}
        record['runs'].append(row)
        for name in ('weighted_path_problems', 'explicit_path_sum_claims', 'claims_with_checkable_error',
                     'arithmetic_sum_errors', 'graph_or_edge_weight_mismatches'):
            aggregate[name] += row[name]
    record['simple_sum_across_runs'] = dict(aggregate)
    return record


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to overwrite')
    args.output.write_text(json.dumps(inspect(), indent=2) + '\n')
