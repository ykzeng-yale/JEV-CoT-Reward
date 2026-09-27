#!/usr/bin/env python3
"""Freeze trusted local training artifacts before prospective task generation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import analyze_screen as screen
from jev_control.controller import fit_controller


def freeze(run, integrity, output):
    if output.exists():
        raise FileExistsError('Frozen artifact destination exists')
    audit_hash = screen.validate_integrity_report(run, integrity)
    local_path = run / 'local_judge.jsonl'
    records = screen.read_jsonl(local_path)
    local = {row['problem_id']: row for row in records}
    if len(local) != len(records):
        raise ValueError('Duplicate local judge problem')
    report = screen.analyze(run, learned=True, local_features=local)
    if report['crossfit']['status'] != 'exploratory_grouped_crossfit':
        raise ValueError('Complete training analysis required')
    ids = [p['problem_id'] for p in report['per_problem']]
    checkpoints = {c['problem_id']: c for c in screen.read_jsonl(run / 'checkpoints.jsonl')}
    rows = screen.read_jsonl(run / 'outcomes.jsonl')
    budget = screen.read_json(run / 'manifest.json')['budget']
    documents, numbers = zip(*(screen.observable_features(checkpoints[p], budget) for p in ids))
    numbers = np.asarray(numbers)
    matrices = {'cheap_tfidf': numbers}
    for kind in ('jev', 'local'):
        semantic = np.vstack([screen.semantic_features(checkpoints[p].get('jev') if kind == 'jev' else local.get(p)) for p in ids])
        if not np.isfinite(semantic).all():
            raise ValueError('Complete semantic training features required for this frozen diagnostic')
        matrices['cheap_tfidf_plus_' + kind] = np.hstack([numbers, semantic])
    controllers = {name: fit_controller(documents, matrix, rows, ids, alpha=10.) for name, matrix in matrices.items()}
    # Independently call the original analysis fit on training inputs to verify
    # exact learner/weighting equivalence. This is a contract check, not accuracy.
    for name, matrix in matrices.items():
        reference, _ = screen.fit_predict(documents, documents, matrix, matrix, rows, ids, ids, 10.)
        np.testing.assert_allclose(controllers[name].predict(documents, matrix), reference, rtol=1e-12, atol=1e-12)
    blob = pickle.dumps(controllers, protocol=5)
    files = [run / name for name in ('manifest.json', 'checkpoints.jsonl', 'outcomes.jsonl', 'local_judge.jsonl', 'local_judge_provenance.json', 'local_judge_status.json')]
    sources = [Path(__file__), Path(screen.__file__), ROOT / 'src/jev_control/controller.py', ROOT / 'src/jev_control/representation.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {'status': 'frozen_before_prospective_generation', 'created_unix': time.time(),
                'training_problem_ids': ids, 'training_run_id': run.name, 'budget': budget,
                'alpha': 10., 'policies': list(controllers), 'actions': list(screen.ACTIONS),
                'cheap_features': list(screen.CHEAP_NAMES), 'semantic_features': list(screen.SEMANTIC_KEYS),
                'integrity_report_sha256': audit_hash, 'training_inputs_sha256': {p.name: digest(p) for p in files},
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
                'artifact_sha256': hashlib.sha256(blob).hexdigest(),
                'security': 'Trusted locally generated pickle only; verify manifest digest before loading. Never load downloaded/untrusted pickle.',
                'training_prediction_equivalence': 'passed at rtol=atol=1e-12; not an evaluation result'}
    output.mkdir(parents=True)
    (output / 'controllers.pkl').write_bytes(blob)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--integrity-report', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    result = freeze(a.run, a.integrity_report, a.output)
    print(json.dumps({'status': result['status'], 'problems': len(result['training_problem_ids']), 'policies': result['policies'], 'artifact_sha256': result['artifact_sha256']}))
