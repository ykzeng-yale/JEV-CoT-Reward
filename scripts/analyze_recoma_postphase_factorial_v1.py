#!/usr/bin/env python3
"""Complete-assignment paired factorial analysis; no survivor rates or generation."""
import argparse,hashlib,json,math
from pathlib import Path

def analyze(design,arms):
    units={t['unit_id']:t for t in design['tasks']};ids={a['arm_id'] for a in design['arms']}
    if len(units)!=24 or len(arms)!=4 or set(arms)!=ids:raise ValueError('complete balanced four-arm assignment required')
    validated={}
    for arm,data in arms.items():
        if data.get('audit_status')!='PASS' or data.get('design_arm_id')!=arm:raise ValueError('independent arm audit binding required')
        if data.get('design_sha256')!=design['_sha256']:raise ValueError('design hash mismatch')
        rows=data['per_task'];by={r['unit_id']:r for r in rows}
        if len(rows)!=len(by) or set(by)!=set(units):raise ValueError('missing, duplicate or unplanned unit')
        for row in rows:
            if type(row['official_success']) is not bool or type(row['scientific_failure']) is not bool:raise ValueError('typed outcome/failure required')
            for k in ['prompt_tokens','generated_tokens','model_calls','environment_actions']:
                if type(row[k]) is not int or row[k]<0:raise ValueError('nonnegative exact generation counters required')
            if not math.isfinite(row['model_service_seconds']) or row['model_service_seconds']<0:raise ValueError('invalid latency')
        validated[arm]=by
    def contrast(high,low):
        diffs={u:int(validated[high][u]['official_success'] and not validated[high][u]['scientific_failure'])-int(validated[low][u]['official_success'] and not validated[low][u]['scientific_failure']) for u in units}
        family={s:sum(v for u,v in diffs.items() if units[u]['scenario']==s)/3 for s in {t['scenario'] for t in units.values()}}
        return {'higher':high,'lower':low,'paired_difference':sum(diffs.values())/24,'wins':sum(v==1 for v in diffs.values()),'losses':sum(v==-1 for v in diffs.values()),'ties':sum(v==0 for v in diffs.values()),'family_differences':family,'population_confidence_interval':None}
    primary=contrast('conservative_90','original_90')
    secondary=[contrast('original_90','original_30'),contrast('conservative_90','conservative_30')]
    return {'status':'COMPLETE_FACTORIAL_DESCRIPTIVE_ANALYSIS','assigned_units':24,'assigned_cells':96,'primary':primary,'secondary_exploratory':secondary,'interaction_difference':secondary[1]['paired_difference']-secondary[0]['paired_difference'],'practical_primary_margin_met':primary['paired_difference']>=0.10,'arm_totals':{a:{k:sum(r[k] for r in rows.values()) for k in ['prompt_tokens','generated_tokens','model_calls','environment_actions','model_service_seconds']} for a,rows in validated.items()},'claim_boundary':'Finite paired development contrasts only; actual arm work may differ despite equal caps. No Jev, transfer, methods novelty or confirmatory population claim. Allocation costs require separate terminal parent inventory.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--design',type=Path,required=True);p.add_argument('--arm-audits',type=Path,nargs=4,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refuse overwrite')
    d=json.loads(a.design.read_text());d['_sha256']=hashlib.sha256(a.design.read_bytes()).hexdigest();arms={}
    for f in a.arm_audits:
        v=json.loads(f.read_text());k=v['design_arm_id']
        if k in arms:raise ValueError('duplicate arm')
        arms[k]=v
    out=analyze(d,arms);out['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.design,*a.arm_audits]};a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
