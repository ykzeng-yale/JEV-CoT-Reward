"""Prospective termination receipts; observations do not override official outcomes."""
import math
ALLOWED={'submit','official_completion','format_failure','action_cap','llm_cap','infrastructure_failure'}
def terminal_receipt(*,trigger,official_card,scientific_failure,environment_actions,model_calls,action_cap,llm_cap):
    if trigger not in ALLOWED:raise ValueError('unknown terminal trigger')
    for value in [environment_actions,model_calls,action_cap,llm_cap]:
        if type(value) is not int or value<0:raise ValueError('typed nonnegative counters required')
    if action_cap==0 or llm_cap==0:raise ValueError('positive caps required')
    if trigger=='infrastructure_failure':
        return {'trigger':trigger,'official_success':None,'outcome_unknown':True,'scientific_failure':scientific_failure,'rerun_or_resample':False}
    if not isinstance(official_card,dict) or type(official_card.get('completed')) is not bool or type(official_card.get('completedSuccessfully')) is not bool:raise ValueError('independent official card required')
    if official_card['completedSuccessfully'] and not official_card['completed']:raise ValueError('inconsistent official success')
    if trigger=='official_completion' and not official_card['completed']:raise ValueError('completion trigger contradicts card')
    if trigger=='format_failure' and not isinstance(scientific_failure,str):raise ValueError('format failure code missing')
    if trigger=='action_cap' and environment_actions<action_cap:raise ValueError('action-cap trigger below cap')
    if trigger=='llm_cap' and model_calls<llm_cap:raise ValueError('LLM-cap trigger below cap')
    return {'trigger':trigger,'official_success':official_card['completedSuccessfully'],'official_completed':official_card['completed'],'failure_adjusted_success':official_card['completedSuccessfully'] and scientific_failure is None,'outcome_unknown':False,'scientific_failure':scientific_failure,'environment_action_cap_reached':environment_actions>=action_cap,'llm_call_cap_reached':model_calls>=llm_cap,'environment_actions':environment_actions,'model_calls':model_calls,'rerun_or_resample':False,'boundary':'Trigger records the executed stop branch. Simultaneous cap observations are retained without claiming an exclusive causal explanation.'}
