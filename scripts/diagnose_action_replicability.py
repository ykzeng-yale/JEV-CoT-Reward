#!/usr/bin/env python3
"""Exploratory split-repeat diagnostic, restricted to the mechanism development release."""
import itertools
import json
from pathlib import Path
from collections import defaultdict
from check_control_release import check

ACTIONS=('continue','repair','branch')

def diagnose(root):
    if root.name!='mechanism-v1-terminal': raise ValueError('Only designated mechanism development data allowed')
    check(root)
    grouped=defaultdict(dict)
    for line in (root/'outcomes.jsonl').read_text().splitlines():
        r=json.loads(line);key=(r['action'],r['repeat'])
        if key in grouped[r['problem_id']]:raise ValueError('Duplicate repeat')
        grouped[r['problem_id']][key]=int(r['outcome']['success'])
    expected={(a,r) for a in ACTIONS for r in range(4)}
    if any(set(v)!=expected for v in grouped.values()):raise ValueError('Need complete three-action four-repeat grid')
    results=[]
    for selected in itertools.combinations(range(4),2):
        held=tuple(r for r in range(4) if r not in selected)
        fit=[];test=[];base=[]
        for rows in grouped.values():
            means=[sum(rows[a,r] for r in selected)/2 for a in ACTIONS]
            winner=ACTIONS[max(range(3),key=lambda j:means[j])]
            fit.append(max(means));test.append(sum(rows[winner,r] for r in held)/2)
            base.append(sum(rows['continue',r] for r in held)/2)
        mean=lambda v:sum(v)/len(v)
        results.append({'selection_repeats':selected,'evaluation_repeats':held,
                        'selection_success':mean(fit),'evaluation_success':mean(test),
                        'continue_evaluation_success':mean(base),'paired_gain':mean(test)-mean(base)})
    return {'status':'exploratory_development_diagnostic','problems':len(grouped),'splits':results,
            'mean_split_gain':sum(r['paired_gain'] for r in results)/len(results),
            'limitations':['Uses outcome-informed checkpoint selection, not a deployable controller.',
                'Split results overlap and are not independent replications or a confidence interval.',
                'Two selection repeats are noisy; failure does not prove absence of action heterogeneity.',
                'The prospective test is excluded; no test outcomes become tuning labels.']}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('release',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    a.output.write_text(json.dumps(diagnose(a.release),indent=2)+'\n')
