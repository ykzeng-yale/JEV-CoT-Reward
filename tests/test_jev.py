import pytest
from concurrent.futures import ThreadPoolExecutor
from jev_control.jev import JevClient, BudgetExceeded, MODEL
from jev_control.features import judge_state


Q = {"check": {"type": "noul", "instructions": "Is this a hypothesis?"}}


def response(payload):
    return {"model": MODEL, "answers": {"check": {"type": "noul", "noul": .6}}, "usage": {"input_tokens": 100}}


def test_durable_cache(tmp_path):
    client = JevClient(tmp_path / "ledger", transport=response, api_key="test")
    a = client.evaluate("public task", Q)
    b = client.evaluate("public task", Q)
    assert not a["cache_hit"] and b["cache_hit"]
    assert client.status()["accounted_usd"] == pytest.approx(.0000042)


def test_timeout_keeps_reservation(tmp_path):
    def fail(payload):
        raise TimeoutError()
    client = JevClient(tmp_path / "ledger", stage_cap_usd=.01, transport=fail, api_key="test")
    with pytest.raises(TimeoutError): client.evaluate("x", Q)
    assert client.status()["accounted_usd"] == .01
    with pytest.raises(BudgetExceeded): client.evaluate("y", Q)


def test_concurrent_hard_cap(tmp_path):
    def fail(payload): raise TimeoutError()
    client = JevClient(tmp_path / "ledger", stage_cap_usd=.03, transport=fail, api_key="test")
    def call(i):
        try: client.evaluate(str(i), Q)
        except (TimeoutError, BudgetExceeded): pass
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(call, range(20)))
    assert client.status()["accounted_usd"] == pytest.approx(.03)


def test_missing_usage_fail_closed(tmp_path):
    client = JevClient(tmp_path / "ledger", transport=lambda p: {}, api_key="test")
    with pytest.raises(ValueError): client.evaluate("x", Q)
    assert client.status()["accounted_usd"] == .01


def test_caps_and_pinning(tmp_path):
    for cap in (26, float("nan"), 0):
        with pytest.raises(ValueError): JevClient(tmp_path / "ledger", stage_cap_usd=cap)
    r = response({}); r["model"] = "jev-latest"
    c = JevClient(tmp_path / "ledger", transport=lambda p:r, api_key="test")
    with pytest.raises(ValueError): c.evaluate("x", Q)


def test_judge_allowlist():
    assert set(judge_state("task", "history", "step")) == {"task", "history", "latest_segment"}

CHOICE = {'check': {'type': 'choice', 'instructions': 'Choose', 'criteria': {'0': 'a', '1': 'b'}}}

def choice_response(payload):
    return {'model': MODEL, 'answers': {'check': {'type': 'choice', 'choice': '1',
            'probabilities': {'0': .2, '1': .8}, 'confidence': .6}}, 'usage': {'input_tokens': 100}}

def test_choice_cache_and_schema(tmp_path):
    c=JevClient(tmp_path/'ledger', transport=choice_response, api_key='test')
    assert c.evaluate('x', CHOICE)['response']['answers']['check']['choice']=='1'
    assert c.evaluate('x', CHOICE)['cache_hit']
    before=c.status()['accounted_usd']
    with pytest.raises(ValueError): c.evaluate('y', {'check': {'type':'choice','instructions':'x'}})
    assert c.status()['accounted_usd']==before

@pytest.mark.parametrize('patch', [
    {'choice':'0'}, {'choice':True}, {'probabilities':{'0':.2,'1':.2}},
    {'probabilities':{'0':float('nan'),'1':.8}}, {'probabilities':{'x':.2,'1':.8}},
    {'confidence':True}, {'confidence':2}])
def test_choice_malformed_retains_reservation(tmp_path,patch):
    r=choice_response(None);r['answers']['check'].update(patch)
    c=JevClient(tmp_path/'ledger', transport=lambda p:r, api_key='test')
    with pytest.raises(ValueError):c.evaluate('x',CHOICE)
    assert c.status()['accounted_usd']==.01
