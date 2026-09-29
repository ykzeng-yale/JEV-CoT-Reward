"""Exploratory parser sensitivity: all standalone integer binary equations per checkpoint."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

PATTERN = re.compile(
    r'(?<![\d+*/×−\-(])(?P<a>\d{1,3})\s*(?P<op>[+\-−*×/])\s*'
    r'(?P<b>\d{1,3})\s*=\s*(?P<c>\d{1,3})(?![\d+*/×−\-])'
)


def value(a, op, b):
    a, b = int(a), int(b)
    if op == '+': return Fraction(a+b)
    if op in ('-', '−'): return Fraction(a-b)
    if op in ('*', '×'): return Fraction(a*b)
    if op == '/':
        return None if b == 0 else Fraction(a, b)
    raise ValueError(op)


def analyze(run, audit_path):
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'passed_arithmetic_horizon_audit':
        raise ValueError('Primary independent audit must pass')
    rows = [json.loads(x) for x in (run/'checkpoints.jsonl').read_text().splitlines() if x]
    op_counts, wrong_op_counts = Counter(), Counter()
    any_error = set(); claim_count = 0
    for row in rows:
        pid = row['problem_id']
        matches = list(PATTERN.finditer(row['state']['latest_segment']))
        for m in matches:
            claim_count += 1
            op = m['op']; op_counts[op] += 1
            got = value(m['a'], op, m['b'])
            correct = got is not None and got == int(m['c'])
            if not correct:
                wrong_op_counts[op] += 1
                any_error.add(pid)
    return {
        'status': 'exploratory_secondary_parser_sensitivity',
        'primary_study_audit_status': audit['status'],
        'eligible_checkpoints': len(rows),
        'standalone_binary_equation_claims': claim_count,
        'claims_by_operator': dict(op_counts),
        'incorrect_claims_by_operator': dict(wrong_op_counts),
        'checkpoints_with_at_least_one_detected_incorrect_claim': len(any_error),
        'primary_frozen_rule_incorrect_first_addition': audit['incorrect'],
        'limitations': [
            'Post-hoc parser sensitivity analysis; it does not change the frozen primary gate.',
            'Only standalone binary integer equations are parsed; chained and compound equations are omitted.',
            'This analysis does not evaluate Jev, correction effects, final task success, or intervention value.',
            'Claims within a checkpoint are dependent; counts are descriptive, not independent-sample inference.',
        ],
    }


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Refuse overwrite')
    a.output.write_text(json.dumps(analyze(a.run,a.audit),indent=2)+'\n')
