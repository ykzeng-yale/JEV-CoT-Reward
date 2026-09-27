# Development v1 gate decision

Date: 2026-09-27. Decision: **PASS_LIMITED_EXPLORATORY_SCREEN**. Proceed with the frozen 24-new-problem mechanism protocol, three actions and four fixed continuations per action. This authorizes a small data-collection stage within the existing research plan; it does not pass the pilot, sequential-control, transfer or Jev-benefit gates.

## Evidence

The [independent audit](../results/development_analysis.json) covers all 48 scheduled episodes from 12 new problems, two seeds and baseline/sham conditions. It reconstructed every call prefix, injected finalization instruction, seed, decoded output and token count, and recomputed strict terminal outcomes. No integrity or outcome disagreement was found. All 24 sham prefixes matched the corresponding baseline prefix before pausing; all reached an eligible online boundary and contained the prespecified surface calculation marker. Marker presence does not establish valid or useful reasoning. Spot-checked graph and arithmetic prefixes contained actual candidate calculations and checks, rather than only input restatement.

| Family | Baseline | Sham |
|---|---:|---:|
| Weighted directed path | 10/12 | 10/12 |
| Six-number arithmetic construction | 2/12 | 2/12 |
| Total | 12/24 | 12/24 |

The mean original-problem difference was zero; the descriptive family-stratified bootstrap range was −12.5 to +12.5 percentage points. This small development sample does not establish equivalence. Outcomes varied within each family, but graph tasks remain easy and arithmetic remains hard. Nine baseline and ten sham arithmetic episodes had invalid final expressions; unfinished reasoning/formatting and number-use errors must remain failures under the frozen endpoint. Do not repair or selectively reparse them into successes.

The run generated 44,933 tokens in 1,053.71 seconds of recorded generation-call service, with a 3.814 GB MLX peak. There were no timeouts. Model loading and orchestration are outside that summed service time. No Jev calls occurred.

## Scope of the next stage

Use task seed 491027, indices 0–23, the same two families and difficulty, the same concise prompt, 1,024 generated tokens per hypothetical episode including its shared initial prefix, and the fixed 96-token final reserve. Capture the first emitted newline at or after 256 tokens, capped at 384, and give judges the last retained paragraph plus prior history. Do not choose checkpoints from Jev scores or outcomes. Keep ineligible problems; do not substitute replacements.

The study asks whether these fixed interventions show useful outcome variation and whether typed features predict it beyond cheap text/statistical features. Preserve family-stratified results; an apparent pooled improvement may merely reflect task-family recognition. The current likelihood branch selector is weak and the generic repair is bounded suffix repair, so a negative result is specific to this action set/model/budget. The screen estimates conditional one-checkpoint behavior, not full sequential deployment.

Jev questions remain batched once per checkpoint, using the central ledger and a **$1 cumulative screen cap within the $25 project ceiling**. The pinned price/model were rechecked against official documentation before launch. No automatic retry, model purchase or scale-up is authorized by a favorable-looking intermediate episode. The full fixed schedule is retained unless a resource or integrity failure interrupts it. A hard interruption with an unmatched generation start has unknown work and must not be analyzed as a complete ledger.

After collection, run the independent local judge on the same states and grouped exploratory analysis. Fewer than 24 complete independent problems blocks the current learned-policy analyzer; report actual enrollment and decide any additional data collection separately. Fresh frozen-policy deployment is still required before claiming Jev changes success.

## Analysis implementation amendments during collection

After the mechanism job started, an independent software audit added stronger source/schedule and local-feature identity checks, plus a training-fold-selected constant action within each task family. The latter addresses the difficulty imbalance already observed in development: recognizing the family must not be mistaken for useful within-family semantic information. These changes do not alter generation, enrollment, outcomes or the frozen rubric. They are disclosed analysis refinements, not a claim that every analysis detail was preregistered. The mechanism stage remains exploratory; fresh policy evaluation is required for an efficacy claim.
