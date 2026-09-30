# Conductivity seeded-random comparator: input-only audit

**Date:** September 30, 2026. This audit reads the frozen protocol and standalone train/dev input census only. It does not inspect `episodes.jsonl`, endpoint fields, calibration labels, success counts, or action-study rates. Reproduce with `scripts/audit_scienceworld_conductivity_random_guess_design.py` and the frozen config/census.

The randomized comparator is the frozen rule

```text
bit = SHA256(initial_full_visible_view_hash + ":" + 20260930) mod 2
```

for the `random_measurement` arm. On the 150 development rows it assigns 68 ones and 82 zeros. Per public target-substance group, the one counts are 10, 14, 11, 13, 10, and 10 among 25 rows each. Both bits occur in every group. There are 147 distinct full visible-input hashes and three repeated-input rows; repeated hashes receive the same bit by construction. All six groups therefore contain individual-level assignment variation, and no group collapses to a single guess.

This is a fixed, reproducible hash baseline with balanced finite-panel assignments. It does **not** establish independent population sampling, label independence, or confirmatory randomization: the bit is a deterministic function of visible inputs, and the panel has only six target-substance groups. It is not an action-success result and does not address Jev. The underlying conductivity action panel remains unauditable while independent runtime replay fails at the first post-action transition; its arm rates remain unopened.

Saved aggregate: `results/scienceworld_conductivity_random_guess_design_audit_v1.json`. The implementation deliberately reads only `input-census.json`; regression tests check exact coverage, duplicate-hash behavior, and use of no endpoint fields.
