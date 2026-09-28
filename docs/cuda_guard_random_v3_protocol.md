# Frozen randomized timing development comparison

September 27, 2026, before generating outcomes. CUDA v2 has 3 entropy-triggered interventions among 55 eligible boundaries (boundary at least 11, full pool capacity), and GUARD and sham both solved 6/12 tasks. This motivates a timing baseline rather than a benefit claim.

Twelve fresh tasks, seed 1291027, same pinned BF16 model, 2048-token total generation allowance, four policies: continue, segmented sham, GUARD adaptation, random branching. Random probability is frozen at 3/55 per eligible visited boundary. Uniform variates are reproducible SHA256(seed:boundary) first-eight-byte fractions. All three candidates and unselected work are charged. Candidate generation/ranking, temperature, reserve and finalization match GUARD. The random schedule is outcome-blind. This is an opportunity-rate-matched baseline in expectation on the development distribution, NOT an exact intervention-count or cost match on new trajectories. Report realized counts/costs and no equivalence claim from identical outcomes.

Primary descriptive contrast: GUARD minus random; also compare each against segmented sham and continue. Problem-level paired analysis, all scheduled and ineligible cases, no replacement or partial-outcome tuning. One repeat is a limited development diagnostic, not a powered efficacy study. Jev is absent; this qualifies control mechanisms before additional semantic acquisition.

One normal-tier pi_fl426 GPU, 4 CPUs, 32 GiB, maximum four hours/14000 cooperative seconds. Existing cached weights only, no API credentials. Full independent token/decision/label audit required. Source and config frozen per run. Earlier records retained; changes to the auditor must not change the historical generation records.
