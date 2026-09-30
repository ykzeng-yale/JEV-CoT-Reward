#!/usr/bin/env python3
"""Reproduce the exact non-outcome prelude before one saved transition."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

from jev_control.scienceworld_conductivity_action_audit import _full_view_hash


def h(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_view(env, protocol, variation):
    env.load(protocol['task'], variationIdx=variation, simplificationStr='', generateGoldPath=False)
    observation, _ = env.reset()
    description = env.taskdescription()
    full = _full_view_hash(env, observation)
    return observation, description, full


def replay_census_prelude(env, protocol, census):
    if len(census) != 450 or [r['variation_id'] for r in census] != list(range(450)):
        raise ValueError('sanitized census prelude is incomplete or reordered')
    for row in census:
        obs, desc, full = load_view(env, protocol, row['variation_id'])
        if h(desc) != row['task_description_sha256'] or full != row['full_sha256']:
            raise ValueError(f"census prelude view mismatch at variation {row['variation_id']}")
    return len(census)


def replay_trace(env, protocol, row):
    obs, _, full = load_view(env, protocol, row['variation_id'])
    if full != row['initial_view_sha256'] or h(obs) != row['initial_look_sha256']:
        return {'initial_match': False, 'actions_replayed': 0, 'first_mismatch': 'initial-view'}
    for i, step in enumerate(row['trace']):
        if step['action'] not in env.get_valid_action_object_combinations():
            return {'initial_match': True, 'actions_replayed': i, 'first_mismatch': {'kind':'action-illegal','index':i}}
        obs, _, _, _ = env.step(step['action'])
        if _full_view_hash(env, obs) != step['view_sha256']:
            return {'initial_match': True, 'actions_replayed': i+1,
                    'first_mismatch': {'kind':'post-action-view','index':i}}
    return {'initial_match': True, 'actions_replayed': len(row['trace']), 'first_mismatch': None}


def main(args):
    protocol=json.loads(args.protocol.read_bytes())
    census=json.loads(args.census.read_bytes())
    integration=json.loads(args.integration_traces.read_bytes())
    target=json.loads(args.target_trace.read_bytes())
    source=args.source_root
    version=source/'scienceworld'/'version.py'
    if not version.exists():
        manifest=ZipFile(source/'scienceworld'/'scienceworld.jar').read('META-INF/MANIFEST.MF').decode()
        spec=next(x.split(': ',1)[1] for x in manifest.splitlines() if x.startswith('Specification-Version:'))
        version.write_text(f'__version__ = {spec!r}\n')
    sys.path.insert(0,str(source))
    from scienceworld import ScienceWorldEnv,__version__
    if __version__ != protocol['expected_scienceworld_version']:
        raise ValueError('pinned runtime mismatch')
    runs=[]
    for rep in range(3):
        env=ScienceWorldEnv(envStepLimit=100)
        try:
            n=replay_census_prelude(env,protocol,census)
            integration_checks=[]
            for row in integration:
                integration_checks.append(replay_trace(env,protocol,row))
            target_obs,_,target_initial=load_view(env,protocol,target['variation_id'])
            target_initial_match=(target_initial==target['initial_view_sha256'] and h(target_obs)==target['initial_look_sha256'])
            action=target['trace'][0]['action']
            if action not in env.get_valid_action_object_combinations():
                post=None
            else:
                post_obs,_,_,_=env.step(action)
                post=_full_view_hash(env,post_obs)
            runs.append({'repetition':rep,'census_rows':n,'training_integration':integration_checks,
                         'target_initial_match':target_initial_match,'first_action':action,
                         'first_post_view_sha256':post,
                         'first_post_matches_record':post==target['trace'][0]['view_sha256']})
        finally:
            env.close()
    result={'variation_id':target['variation_id'],'policy':target['policy'],'runs':runs,
            'prelude_exact':all(r['census_rows']==450 and all(x['first_mismatch'] is None for x in r['training_integration']) for r in runs),
            'target_initial_all_match':all(r['target_initial_match'] for r in runs),
            'post_hashes_stable':len({r['first_post_view_sha256'] for r in runs})==1,
            'post_hash_all_match_record':all(r['first_post_matches_record'] for r in runs),
            'endpoint_or_outcome_fields_staged_or_read':False}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:result[k] for k in ('variation_id','policy','prelude_exact','target_initial_all_match','post_hashes_stable','post_hash_all_match_record','endpoint_or_outcome_fields_staged_or_read')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','census','integration-traces','target-trace','source-root','output'):
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    main(args)
