#!/usr/bin/env python3
"""Locate the first visible-state divergence in one fixed failed audit row."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

from jev_control.scienceworld_conductivity_action_audit import (
    _full_view_hash, _replay_census, evaluate_terminal,
)


def main(protocol_path, result_path, source_root, output_path, variation, policy):
    protocol = json.loads(protocol_path.read_bytes())
    result = json.loads(result_path.read_bytes())
    record = next(row for row in result['episodes']
                  if row['variation_id'] == variation and row['policy'] == policy)
    version_file = source_root / 'scienceworld' / 'version.py'
    if not version_file.exists():
        manifest = ZipFile(source_root / 'scienceworld' / 'scienceworld.jar').read('META-INF/MANIFEST.MF').decode('utf-8')
        version = next(line.split(': ', 1)[1] for line in manifest.splitlines()
                       if line.startswith('Specification-Version:'))
        version_file.write_text(f'__version__ = {version!r}\n')
    sys.path.insert(0, str(source_root))
    from scienceworld import ScienceWorldEnv, __version__
    if __version__ != protocol['expected_scienceworld_version']:
        raise ValueError('pinned runtime mismatch')
    env = ScienceWorldEnv(envStepLimit=100)
    try:
        census_rows = _replay_census(env, protocol, result)
        env.load(protocol['task'], variationIdx=variation, simplificationStr='', generateGoldPath=False)
        observation, _ = env.reset()
        target_match = re.search(r'Your task is to determine if (unknown substance [A-Z]) is electrically conductive\.', env.taskdescription())
        checks = {
            'variation_id': variation,
            'policy': policy,
            'census_rows_replayed': census_rows,
            'initial_full_view_match': _full_view_hash(env, observation) == record['initial_view_sha256'],
            'initial_look_match': hashlib.sha256(observation.encode()).hexdigest() == record['initial_look_sha256'],
            'task_description_target_match': target_match is not None and target_match.group(1) == record['target_group'],
            'actions_replayed': 0,
            'first_divergence': None,
            'reading_digest_match': None,
            'terminal_endpoint_match': None,
        }
        reading_digest = None
        if not checks['initial_full_view_match'] or not checks['initial_look_match'] or not checks['task_description_target_match']:
            checks['first_divergence'] = 'initial-view-or-description'
        else:
            for index, step in enumerate(record['trace']):
                legal = env.get_valid_action_object_combinations()
                if step['action'] not in legal:
                    checks['first_divergence'] = {'kind': 'recorded-action-not-currently-legal', 'action_index': index}
                    break
                observation, _, _, _ = env.step(step['action'])
                checks['actions_replayed'] += 1
                got = _full_view_hash(env, observation)
                if step['action'].startswith(('look at ', 'examine ')):
                    reading_digest = hashlib.sha256(observation.encode()).hexdigest()
                if got != step['view_sha256']:
                    checks['first_divergence'] = {'kind': 'post-action-visible-view-hash', 'action_index': index}
                    break
            if checks['first_divergence'] is None:
                checks['reading_digest_match'] = reading_digest == record['reading_sha256']
                actual = evaluate_terminal(env, record['target_group'], record['chosen_box'], record['prediction'])
                checks['terminal_endpoint_match'] = actual == record['endpoint']
                if checks['reading_digest_match'] is False:
                    checks['first_divergence'] = 'measurement-reading-digest'
                elif checks['terminal_endpoint_match'] is False:
                    checks['first_divergence'] = 'terminal-endpoint-flags'
        output_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + '\n')
        print(json.dumps({k: checks[k] for k in ('variation_id','policy','census_rows_replayed','actions_replayed','first_divergence','reading_digest_match','terminal_endpoint_match')}))
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','result','source-root','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--variation',type=int,required=True)
    p.add_argument('--policy',choices=['measurement','masked_measurement','random_measurement','continue_prior'],required=True)
    a=p.parse_args(); main(a.protocol,a.result,a.source_root,a.output,a.variation,a.policy)
