import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('receipt',Path(__file__).parents[1]/'scripts/recoma_terminal_receipt_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def args():return dict(trigger='submit',official_card={'completed':False,'completedSuccessfully':False},scientific_failure=None,environment_actions=4,model_calls=4,action_cap=30,llm_cap=62)
def test_submit_text_cannot_create_official_success():
 r=m.terminal_receipt(**args());assert r['official_success'] is False and not r['failure_adjusted_success']
def test_actual_completion_is_distinct_from_submission():
 a=args();a.update(trigger='official_completion',official_card={'completed':True,'completedSuccessfully':True});assert m.terminal_receipt(**a)['failure_adjusted_success']
def test_format_failure_retains_card_but_adjusts_success():
 a=args();a.update(trigger='format_failure',scientific_failure='missing_action_json',official_card={'completed':True,'completedSuccessfully':True});r=m.terminal_receipt(**a);assert r['official_success'] and not r['failure_adjusted_success']
def test_simultaneous_caps_not_exclusive_cause():
 a=args();a.update(trigger='submit',environment_actions=30,model_calls=62);r=m.terminal_receipt(**a);assert r['trigger']=='submit' and r['environment_action_cap_reached'] and r['llm_call_cap_reached']
def test_infrastructure_failure_never_zero_imputes_endpoint():
 a=args();a.update(trigger='infrastructure_failure',official_card=None);r=m.terminal_receipt(**a);assert r['official_success'] is None and r['outcome_unknown']
@pytest.mark.parametrize('change',[{'trigger':'official_completion'},{'trigger':'action_cap'},{'trigger':'llm_cap'},{'trigger':'format_failure'},{'model_calls':True},{'official_card':{'completed':False,'completedSuccessfully':True}}])
def test_contradictory_or_untyped_evidence_rejected(change):
 a=args();a.update(change)
 with pytest.raises(ValueError):m.terminal_receipt(**a)
