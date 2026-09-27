"""Malformed-response and preflight regressions; no paid calls."""
import pytest

from jev_control.jev import BudgetExceeded, HARD_CAP_USD, JevClient, MODEL


Q = {"check": {"type": "noul", "instructions": "Is this a hypothesis?"}}


def response(value):
    return {"model": MODEL, "answers": {"check": {"type": "noul", "noul": value}},
            "usage": {"input_tokens": 100}}


@pytest.mark.parametrize("value", [True, False, None, "0.5", float("nan"), 1.1])
def test_invalid_noul_retains_reservation(tmp_path, value):
    c = JevClient(tmp_path / "ledger", transport=lambda p: response(value), api_key="test")
    with pytest.raises(ValueError):
        c.evaluate("public", Q)
    assert c.status()["accounted_usd"] == .01


@pytest.mark.parametrize("bad", [[], None, {"usage": []}, {"usage": {"input_tokens": 100}, "model": MODEL, "answers": []}])
def test_malformed_response_fails_uniformly(tmp_path, bad):
    c = JevClient(tmp_path / "ledger", transport=lambda p: bad, api_key="test")
    with pytest.raises(ValueError):
        c.evaluate("public", Q)
    assert c.status()["accounted_usd"] == .01


def test_unsupported_question_rejected_before_network_or_reservation(tmp_path):
    calls = []
    c = JevClient(tmp_path / "ledger", transport=lambda p: calls.append(p), api_key="test")
    with pytest.raises(ValueError):
        c.evaluate("public", {"check": {"type": "choice", "instructions": "Choose."}})
    assert not calls
    assert c.status()["accounted_usd"] == 0


def test_nonfinite_payload_rejected_before_reservation(tmp_path):
    c = JevClient(tmp_path / "ledger", transport=lambda p: response(.5), api_key="test")
    with pytest.raises(ValueError):
        c.evaluate({"bad": float("nan")}, Q)
    assert c.status()["accounted_usd"] == 0


def test_excess_usage_freezes_even_if_answer_is_malformed(tmp_path):
    bad = {"model": MODEL, "usage": {"input_tokens": 10**1000}, "answers": []}
    c = JevClient(tmp_path / "ledger", stage_cap_usd=25, transport=lambda p: bad, api_key="test")
    with pytest.raises(RuntimeError, match="reservation"):
        c.evaluate("public", Q)
    assert c.status()["accounted_usd"] == HARD_CAP_USD
    with pytest.raises(BudgetExceeded):
        c.evaluate("another", Q)
