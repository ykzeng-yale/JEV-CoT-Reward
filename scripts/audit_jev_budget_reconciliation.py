"""Offline cost-only reconciliation; never instantiate JevClient or read responses.

The private SQLite database is opened read-only. Public output contains aggregates,
not request IDs, cache keys, payloads, response values, credentials or task outcomes.
Historical snapshots are compared to settled chronological prefixes, not summed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

PRICE_PER_MILLION = 0.042
RESERVATION_USD = 0.01
HARD_CAP_USD = 25.0
DEFAULT_DB = Path.home() / '.local/state/jev-cot-reward/jev.sqlite3'
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = (
    ('results/jev_budget.json', ()),
    ('runs/phase0-20260927/summary.json', ('jev_budget',)),
    ('runs/phase0-budget1024-20260927/summary.json', ('jev_budget',)),
    ('runs/mechanism-v1-20260927/summary.json', ('jev_budget',)),
    ('runs/branch-replication-v3a-20260927/manifest.json', ('jev_ledger_before',)),
    ('runs/branch-replication-v3a-20260927/manifest.json', ('jev_ledger_after',)),
    ('runs/jev-arithmetic-pairs-v1-20260928/status.json', ('ledger_before',)),
    ('runs/jev-arithmetic-pairs-v1-20260928/status.json', ('ledger_after',)),
    ('runs/jev-natural-addition-v1-20260928/status.json', ('ledger_before',)),
    ('runs/jev-natural-addition-v1-20260928/status.json', ('ledger_after',)),
)


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def total(rows):
    return math.fsum(row[1] for row in rows)


def groups(rows):
    parts = defaultdict(list)
    for row in rows:
        parts[row[0]].append(row[1])
    return [{'status': name, 'attempts': len(values), 'usd': math.fsum(values)}
            for name, values in sorted(parts.items())]


def read_ledger(path):
    # mode=ro prevents creation/writes; authorizer prevents reading private responses.
    path = Path(path).resolve(strict=True)
    allowed = {'status', 'charged_usd', 'input_tokens', 'started', 'cache_key', 'ROWID'}
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as con:
        con.execute('PRAGMA query_only=ON')
        con.execute('BEGIN')
        integrity = con.execute('PRAGMA integrity_check').fetchone()[0]
        con.set_authorizer(lambda action, table, column, *_:
            sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == 'requests'
            and column not in allowed else sqlite3.SQLITE_OK)
        rows = con.execute('SELECT status, charged_usd, input_tokens, started '
                           'FROM requests ORDER BY started, rowid').fetchall()
        duplicates = con.execute("SELECT COUNT(*) FROM (SELECT cache_key FROM requests "
                                 "WHERE status IN ('ok','reserved') GROUP BY cache_key "
                                 'HAVING COUNT(*)>1)').fetchone()[0]
    return rows, integrity, duplicates


def reconcile_snapshot(rows, snapshot):
    count = sum(g['attempts'] for g in snapshot['groups'])
    prefix = rows[:count]
    expected = {g['status']: g for g in groups(prefix)}
    actual = {g['status']: g for g in snapshot['groups']}
    matches = (count <= len(rows) and len(actual) == len(snapshot['groups']) and
               set(expected) == set(actual) and
               close(snapshot['accounted_usd'], total(prefix)) and
               close(snapshot['accounted_usd'], math.fsum(g['usd'] for g in actual.values())) and
               all(actual[k]['attempts'] == expected[k]['attempts'] and
                   close(actual[k]['usd'], expected[k]['usd']) for k in expected))
    return {'attempts': count, 'accounted_usd': snapshot['accounted_usd'],
            'matches_settled_chronological_prefix': matches,
            'is_current_by_attempt_count': count == len(rows),
            'later_attempts': max(0, len(rows) - count),
            'later_accounted_usd': total(rows[count:]) if matches else None}


def audit(root, db):
    rows, integrity, duplicates = read_ledger(db)
    errors = []
    if integrity != 'ok': errors.append('SQLite integrity check failed')
    if duplicates: errors.append('Duplicate active/successful cache entries')
    for status, usd, tokens, started in rows:
        if status not in {'ok', 'reserved', 'unresolved', 'price_anomaly'}:
            errors.append('Unknown ledger status')
        if not math.isfinite(usd) or usd < 0 or not math.isfinite(started):
            errors.append('Nonfinite or negative monetary/time metadata')
        if status == 'ok' and (type(tokens) is not int or tokens < 0 or
                              not close(usd, tokens * PRICE_PER_MILLION / 1e6)):
            errors.append('Successful row does not match pinned input-token price')
        if status in {'reserved', 'unresolved'} and not close(usd, RESERVATION_USD):
            errors.append('Unsettled row does not retain the configured reservation')
    root = Path(root)
    snapshots = []
    for relative, pointer in SNAPSHOTS:
        path = root / relative
        if not path.exists(): continue
        metadata = json.loads(path.read_text())
        for name in pointer: metadata = metadata[name]
        check = reconcile_snapshot(rows, metadata)
        snapshots.append({'source': relative, 'metadata_key': '.'.join(pointer) or '<root>', **check})
        if not check['matches_settled_chronological_prefix']:
            errors.append(f'Historical snapshot prefix requires review: {relative}:{pointer}')
    ok = [r for r in rows if r[0] == 'ok']
    unresolved = [r for r in rows if r[0] == 'unresolved']
    run_cost_checks = []
    mechanism = root / 'results/mechanism_v1_analysis.json'
    first = next((s for s in snapshots if s['source'] == 'results/jev_budget.json'), None)
    after = next((s for s in snapshots if s['source'] == 'runs/mechanism-v1-20260927/summary.json'), None)
    if mechanism.exists() and first and after:
        # Read only this named cost scalar into the report; no outcome fields are used.
        declared = json.loads(mechanism.read_text())['costs']['jev_new_call_usd_in_this_run']
        delta = total(rows[first['attempts']:after['attempts']])
        matches = close(declared, delta)
        run_cost_checks.append({'source': 'results/mechanism_v1_analysis.json',
                               'metadata_key': 'costs.jev_new_call_usd_in_this_run',
                               'declared_new_call_usd': declared,
                               'matching_cumulative_ledger_delta_usd': delta,
                               'matches': matches})
        if not matches: errors.append('Mechanism run cost does not match cumulative-ledger increment')
    metadata = json.dumps(rows, separators=(',', ':'), allow_nan=False).encode()
    iso = lambda value: datetime.fromtimestamp(value, timezone.utc).isoformat()
    report = {
        'schema_version': 1, 'audit_status': 'passed_cost_metadata_reconciliation' if not errors else 'review_required',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'ledger': {'locator': '~/.local/state/jev-cot-reward/jev.sqlite3',
                   'read_mode': 'SQLite mode=ro; query_only; one read transaction; response-column access denied',
                   'cost_metadata_sha256': hashlib.sha256(metadata).hexdigest(),
                   'integrity_check': integrity, 'attempts': len(rows), 'groups': groups(rows),
                   'first_attempt_utc': iso(rows[0][3]) if rows else None,
                   'last_attempt_utc': iso(rows[-1][3]) if rows else None,
                   'successful_input_tokens': sum(r[2] for r in ok),
                   'successful_token_priced_usd': total(ok),
                   'unresolved_reserved_usd': total(unresolved), 'accounted_usd': total(rows),
                   'hard_cap_usd': HARD_CAP_USD, 'within_hard_cap': total(rows) <= HARD_CAP_USD,
                   'duplicate_active_or_successful_cache_entries': duplicates},
        'historical_snapshots': snapshots, 'run_cost_checks': run_cost_checks,
        'errors': sorted(set(errors)),
        'interpretation': [
            'A matching smaller snapshot is historical; it is not an additional charge or the live cumulative total.',
            'Successful cost is calculated from recorded input-token usage and the pinned adapter price; no provider invoice was checked.',
            'Unresolved reservations are conservative accounted exposure, not verified invoice charges.',
            'Cache-hit reuse incurs no new ledger row. Per-policy uncached-equivalent costs must not be summed as actual spend.',
            'Historical prefix matching assumes relevant attempts were settled when snapshots were saved; later status transitions can require review.',
            'This audit covers the configured central local ledger, not proof that every external/provider account request used that ledger.',
            'No API calls, private response-column reads, credentials, task labels or outcome analysis are used.'
        ]}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.db)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'audit_status': result['audit_status'], 'attempts': result['ledger']['attempts'],
                      'successful_token_priced_usd': result['ledger']['successful_token_priced_usd'],
                      'unresolved_reserved_usd': result['ledger']['unresolved_reserved_usd'],
                      'accounted_usd': result['ledger']['accounted_usd'], 'errors': result['errors']}))
    raise SystemExit(0 if not result['errors'] else 1)
