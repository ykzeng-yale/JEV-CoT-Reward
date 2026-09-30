# Jev cost reconciliation — September 30, 2026

The claimed discrepancy between `results/jev_budget.json` ($0.000223734) and the mechanism study's new-call cost ($0.001022406) is explained by a **historical exported snapshot**. It is not evidence of a broken live cumulative ledger. The historical file exactly matches the first seven settled requests; the mechanism study brought that count to 31 and cumulative cost to $0.001246140. Subtracting the seven-request snapshot gives exactly $0.001022406, matching the study's cost metadata. Preserve the original snapshot as historical provenance.

The independent, offline, read-only audit of the configured central ledger returned:

| Quantity | Verified value |
| --- | ---: |
| Total ledger attempts | 143 |
| Successful requests | 142 |
| Recorded successful input tokens | 135,969 |
| Successful input-token-priced cost | $0.005710698 |
| Unresolved attempts | 1 |
| Conservative unresolved reservation | $0.010000000 |
| Cumulative accounted amount | **$0.015710698** |
| Hard cumulative cap | $25.00 |
| Last recorded request | 2026-09-29 00:16:47 UTC |

All ten inspected before/after or exported cost snapshots match settled chronological prefixes of the central ledger. SQLite integrity passed, every successful row matches the adapter's pinned $0.042/million-input-token price, the unresolved row retains its $0.01 reservation, and no duplicate active/successful cache entry was found. The latest natural-addition status snapshot matches the current ledger exactly. Its 48 successful evaluations include 14 cache hits; only 34 new successful ledger requests were added, costing $0.001131480. Historical snapshots, reuse costs and per-policy uncached-equivalent costs must not be added together as actual provider spending.

This corrects the prior assessment's characterization of an unresolved cumulative-accounting discrepancy. A stale public export should not itself block the study. The separately documented scientific and TypeSafe-terms decisions remain separate questions; this audit makes no new API request and does not grant new authorization.

Reproduction:

```sh
.venv/bin/python scripts/audit_jev_budget_reconciliation.py \
  --output results/jev_budget_reconciliation_20260930.json
.venv/bin/python -m pytest tests/test_jev_budget_reconciliation.py tests/test_jev.py tests/test_jev_review.py -q
```

Validation: 34 tests passed, including historical-prefix matching, failed-attempt reservation retention, corrupt-price rejection, duplicate-cache detection, non-creation of missing databases and unchanged database bytes after audit. The auditor opens SQLite with `mode=ro`, enables `query_only`, uses one read transaction and denies response-column reads; it never constructs `JevClient`. No credentials, API calls, private response values, task labels or outcome analysis are used.

Audit limits: successful cost is a reconstruction from recorded provider input-token usage at the pinned adapter price, not a provider invoice. The unresolved $0.01 is conservative exposure, not a verified charge. A matching ledger does not prove that no external request ever bypassed it. Historical-prefix matching assumes the selected snapshots were taken after relevant requests settled; later status transitions can require review. This audit does not reopen the closed v1 evaluation and does not alter the private database, old snapshots or budget limits.
