#!/usr/bin/env python3
"""Independent exact-protocol and runtime replay audit before rate analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from jev_control.scienceworld_conductivity_action_audit import audit


def main(protocol_path, result_path, source_root, output):
    raw = protocol_path.read_bytes()
    protocol, result = json.loads(raw), json.loads(result_path.read_bytes())
    if result['protocol_sha256'] != hashlib.sha256(raw).hexdigest() or result['protocol'] != protocol['protocol']:
        raise ValueError('exact frozen protocol mismatch')
    if any(result[key] != 0 for key in ('test_loaded', 'model_calls', 'jev_calls')) or result['gold_paths_requested'] is not False:
        raise ValueError('exclusion contract violated')
    if result['controller_fields'] != ['task_description', 'observation', 'legal_actions']:
        raise ValueError('controller input contract mismatch')
    for rec in protocol['source_members'].values():
        file = source_root / rec['path'].removeprefix('scienceworld-source/')
        if hashlib.sha256(file.read_bytes()).hexdigest() != rec['sha256']:
            raise ValueError('independent source hash mismatch')
    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__
    if __version__ != protocol['expected_scienceworld_version']:
        raise ValueError('independent runtime version mismatch')
    report = audit(protocol, result, env_factory=lambda: ScienceWorldEnv(envStepLimit=100))
    report['result_sha256'] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    report['protocol_sha256'] = hashlib.sha256(raw).hexdigest()
    with output.open('x') as f:
        f.write(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passes': report['passes'], 'replay_episodes_verified': report['replay_episodes_verified']}))
    return 0 if report['passes'] else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('protocol', 'result', 'source-root', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(main(a.protocol, a.result, a.source_root, a.output))
