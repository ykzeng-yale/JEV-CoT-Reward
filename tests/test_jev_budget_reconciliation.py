import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest
from jev_control import jev

spec = importlib.util.spec_from_file_location('budget_reconciliation', Path(__file__).parents[1] / 'scripts/audit_jev_budget_reconciliation.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def database(tmp_path, rows):
    path = tmp_path / 'ledger.sqlite3'
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE requests (status TEXT, charged_usd REAL, input_tokens INTEGER, started REAL, cache_key TEXT, response TEXT)')
        for i, row in enumerate(rows):
            con.execute('INSERT INTO requests VALUES (?,?,?,?,?,?)', (*row, str(i), 'PRIVATE_RESPONSE_MUST_NOT_APPEAR'))
    return path


def test_historical_snapshot_and_reservations_are_not_double_counted(tmp_path):
    rows = [('ok', .0000042, 100, 1.), ('ok', .0000084, 200, 2.), ('unresolved', .01, None, 3.)]
    path = database(tmp_path, rows)
    (tmp_path/'results').mkdir()
    (tmp_path/'results/jev_budget.json').write_text(json.dumps({'accounted_usd': .0000042, 'groups': [{'status':'ok','attempts':1,'usd':.0000042}]}))
    before = path.read_bytes()
    result = audit.audit(tmp_path, path)
    assert result['audit_status'] == 'passed_cost_metadata_reconciliation'
    assert result['ledger']['successful_input_tokens'] == 300
    assert result['ledger']['accounted_usd'] == pytest.approx(.0100126)
    assert result['historical_snapshots'][0]['matches_settled_chronological_prefix']
    assert result['historical_snapshots'][0]['later_attempts'] == 2
    assert not result['historical_snapshots'][0]['is_current_by_attempt_count']
    assert path.read_bytes() == before
    assert 'PRIVATE_RESPONSE' not in json.dumps(result)


def test_reject_corrupt_successful_price(tmp_path):
    path = database(tmp_path, [('ok', .0001, 100, 1.)])
    result = audit.audit(tmp_path, path)
    assert result['audit_status'] == 'review_required'
    assert 'Successful row does not match pinned input-token price' in result['errors']


def test_inconsistent_snapshot_does_not_masquerade_as_stale(tmp_path):
    rows = [('ok', .0000042, 100, 1.)]
    result = audit.reconcile_snapshot(rows, {'accounted_usd':.1, 'groups':[{'status':'ok','attempts':1,'usd':.1}]})
    assert not result['matches_settled_chronological_prefix']
    assert result['later_accounted_usd'] is None


def test_missing_database_is_never_created(tmp_path):
    path = tmp_path/'nonexistent.sqlite3'
    with pytest.raises(FileNotFoundError): audit.read_ledger(path)
    assert not path.exists()


def test_unresolved_reservation_must_not_disappear(tmp_path):
    path = database(tmp_path, [('unresolved', 0, None, 1.)])
    result = audit.audit(tmp_path, path)
    assert 'Unsettled row does not retain the configured reservation' in result['errors']


def test_duplicate_success_cache_detected(tmp_path):
    path = database(tmp_path, [('ok', .0000042, 100, 1.), ('ok', .0000042, 100, 2.)])
    with sqlite3.connect(path) as con: con.execute("UPDATE requests SET cache_key='same'")
    result = audit.audit(tmp_path, path)
    assert result['ledger']['duplicate_active_or_successful_cache_entries'] == 1
    assert result['audit_status'] == 'review_required'


def test_audit_constants_match_current_adapter():
    assert audit.PRICE_PER_MILLION == jev.PRICE_PER_MILLION
    assert audit.RESERVATION_USD == jev.RESERVE_USD
    assert audit.HARD_CAP_USD == jev.HARD_CAP_USD
