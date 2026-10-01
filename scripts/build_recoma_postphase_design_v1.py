#!/usr/bin/env python3
"""Prepare a release-gated factorial development design; never submit or generate."""
import argparse, hashlib, json
from pathlib import Path
CONSERVATIVE = ('Before SUBMIT, review the visible observations and task instructions. '
                'Do not infer completion from your intention or a successful tool call alone. '
                'If visible evidence is insufficient, continue gathering task-relevant evidence '
                'within the remaining action budget. SUBMIT remains available.')
def build(manifest, analysis):
    assert analysis['analysis_status']=='COMPLETE_AUDITED_PANEL'
    assert analysis['assigned_denominator']==24 and analysis['frozen_seeds']==[0]
    strata=sorted({(r['scenario'],r['difficulty']) for r in analysis['per_task']})
    assert len(strata)==24 and len({s for s,d in strata})==8
    arms=[{'arm_id':f'{prompt}_{cap}','prompt':prompt,'environment_action_cap':cap,
           'additional_instruction':CONSERVATIVE if prompt=='conservative' else '',
           'llm_call_cap':2*cap+2} for prompt in ['original','conservative'] for cap in [30,90]]
    tasks=[{'scenario':s,'difficulty':d,'seed':1,'unit_id':f'{s}_{d}_1'} for s,d in strata]
    cells=[{'unit_id':t['unit_id'],'arm_id':a['arm_id']} for t in tasks for a in arms]
    assert len(cells)==96 and len({(x['unit_id'],x['arm_id']) for x in cells})==96
    return {'schema':'recoma_postphase_design_v1','release_status':'PREPARED_NOT_RELEASED',
      'generation_authorized_by_this_file':False,'launch_blockers':['documented post-phase compute envelope','implementation and terminal-reason receipt validation','fresh-data provenance gate','independent prospective protocol review'],
      'task_role':'fresh-seed development; shared families are not independent transfer',
      'model':manifest['model'],'model_revision':manifest['model_revision'],
      'context_ceiling':32768,'output_token_cap':400,'tasks':tasks,'arms':arms,'cells':cells,
      'primary_contrast':{'higher':'conservative_90','lower':'original_90','endpoint':'official_success','practical_margin':0.10},
      'secondary_contrasts':['original_90-original_30','conservative_90-conservative_30','prompt-by-horizon interaction'],
      'analysis':'All assigned units, paired task differences; eight-family descriptive sensitivity only. No unseen-family confidence or pooled MLX/CUDA inference.',
      'multiplicity':'One primary contrast; other contrasts exploratory, no success-based switching.',
      'fairness':'Primary prompt contrast has equal caps, but actual tokens/actions/latency may differ. Horizon contrasts deliberately have unequal budgets and cannot establish matched-cost superiority.',
      'go_rule':'Practical primary margin alone does not qualify Jev: require useful action choices, cheap/local comparator headroom and fresh confirmation before judge study.',
      'no_go_rule':'Zero success in every arm or no practical baseline improvement stops this DiscoveryWorld operating regime, not all possible settings.',
      'terminal_receipt_contract':['valid SUBMIT','official completion','typed format failure','action cap','LLM cap','infrastructure failure; ambiguous causes remain ambiguous'],
      'cost_contract':'All parent allocations and setup/idle/failures; all prefill/generated/discarded work; no Jev or hosted calls; actual cost mismatch reported.',
      'seed0_outcomes_used_only_for_design':True,'unseen_final_evaluation_opened':False}
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('refuse overwrite')
    d=build(json.loads(a.manifest.read_text()),json.loads(a.analysis.read_text()))
    d['design_input_sha256']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.manifest,a.analysis]}
    a.output.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
