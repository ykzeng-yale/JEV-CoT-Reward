# Literature and implementation audit

Audit date: 2026-09-27. This report checks primary papers and selected public source files; it does **not** reproduce their benchmark results. Repository commits and downloaded-file hashes are recorded in `source_manifest.json`. “Verified” means the specified material was inspected, not that its empirical claims were independently established.

## Decision for this project

The earlier proposal misses two especially close papers: **Calibration Is Not Control** and **DIAL**. They already study intervention value rather than failure probability and learn control from matched counterfactual continuations. Consequently, “outcome-grounded checkpoint intervention learning” is a sound experimental foundation but is **not a defensible new conceptual contribution by itself**. Jev-Mem also already uses Jev as a controller in an agent system.

Proceed as a **replication and carefully isolated extension**. The useful empirical question is whether a frozen, inexpensive typed semantic judge adds enough *decision value* beyond strong local features to justify its acquisition cost, particularly when the receiver, task family, or horizon changes. Selective judge acquisition is an additional testable mechanism, not an established novelty claim. A positive pilot would justify a second, focused novelty search before writing a methods-paper claim.

The next experiment should retain the same generator, intervention definitions, branching selector, budget accounting, and outcome verifier across feature conditions. Compare cheap statistics **plus text representations**, a local typed judge, and Jev. Otherwise it is too easy to attribute ordinary representation or controller improvements to Jev.

## Closest omitted prior art

### Calibration Is Not Control: Why LLM-Agent Oversight Needs Intervention

**Verified paper:** [arXiv:2606.21399v1](https://arxiv.org/html/2606.21399v1), submitted June 19, 2026; especially §§2–4 and Appendix B. It defines intervention advantage, characterizes when a scalar supports optimal control, and measures abstraction loss. Its prefix-branching protocol replays an identical state and executes continue, a task-specific intervention, and quit. A random forest predicts per-action success; a variance penalty makes selection conservative. Evaluations include ALFWorld, ScienceWorld, GSM8K, and HotpotQA, with trajectory-grouped splits and some repeated branching.

This is a **direct novelty blocker** for the proposed estimand, counterfactual collection protocol, and scalar-versus-action-value pitch. The paper itself limits its main evidence to offline prefix evaluation and calls for richer online intervention policies. That leaves room for measured extensions, but simply adding a third action or a random forest does not distinguish our proposal. I did not locate an official code link in the paper or title search; any baseline built from it should be labeled a paper-based reimplementation.

### DIAL: Same Signal, Opposite Meaning

**Verified paper:** [arXiv:2605.06908v1](https://arxiv.org/html/2605.06908v1), submitted May 7, 2026; §§3–4, limitations, and Appendix C. DIAL distinguishes computation need from computation suitability. It randomly selects intervention checkpoints independently of the signal being studied, forks the environment, compares base and optimizer actions using truncated continuations, and fits a sparse logistic gate. Universal features include step count, entropy, evidence, and available actions; an LLM proposes extra task-specific features. Gates are learned per environment/backbone. The paper also discusses refresh under drift.

This blocks claims that random checkpoint exploration, utility-direction learning, semantic feature proposals, or checking reversal across models are new. Its deployment gate avoids repeated LLM calls, making it a serious cost comparator for Jev. A possible distinction is *whether the same predefined typed schema transfers with fewer destination outcomes*, versus learning a separate feature/gate recipe for each setting. That is a hypothesis to test. No official implementation was verified in the inspected paper or title search.

### AERA: Adaptive Evidence Residual Allocation

**Verified paper:** [arXiv:2608.27964v1](https://arxiv.org/html/2608.27964v1), submitted August 28, 2026. AERA learns whether more response sampling can recover a better answer. Its checkpoints represent accumulated response populations, not successive tokens of one CoT. Features include answer distribution, temporal change, re-solving, semantics, and cost. Offline labels use future correctness, while deployment only sees the current response pool. Its tested binary label means currently wrong but correct at some later checkpoint; the authors explicitly separate that special case from general graded utility.

This blocks “learn future compute value instead of confidence” as a broad contribution. It is less directly matched to local repair versus branching within one trace. If our actions include stopping or allocating samples, AERA belongs in the comparison. No official code link was verified in the inspected paper.

### CausalFlow

**Verified paper, secondary-depth read:** [arXiv:2605.25338v1](https://arxiv.org/html/2605.25338v1), submitted May 25, 2026. It performs step interventions on failed traces to identify responsibility and construct validated minimal repairs, supporting test-time recovery and training supervision. This further rules out treating “execute counterfactual repairs and learn from the results” as an unoccupied area. Its focus is localization and repair, rather than the incremental value of acquiring a cheap semantic observation before selecting among budget-matched actions. Full implementation and empirical reproduction remain unverified in this audit.

## Direct Jev evidence

### JEV-as-a-Judge

**Verified paper:** [arXiv:2609.26550v1](https://arxiv.org/html/2609.26550v1), submitted September 22, 2026. It compares Jev with generative and reward-model judges and evaluates frozen confidence cascades. Stronger limitations appear for derivation checking and persuasive incorrect responses. In its small reference-free prose follow-up, Jev's error AUROC is 0.518 over 80 responses; that is a workload-specific result, not a universal measurement of Jev. The paper explicitly treats calibration as task/interface-dependent and reports failures of threshold transfer.

This is evidence for evaluating Jev as an inexpensive sensor or first-stage judge. It does not demonstrate a trained reasoning-intervention controller, a guarantee for pruning, or improvement from optimizing a generator against Jev scores. Its cascade is also prior art for simply escalating uncertain Jev calls.

### Just Ask Jev / RLCDAlignBench

**Verified paper:** [arXiv:2609.29429v1](https://arxiv.org/html/2609.29429v1), submitted September 24, 2026. It evaluates detection across 44 benchmarks and ten alignment-failure categories. The generic-question median AUROC of 0.886 concerns detection labels, not downstream intervention effectiveness. Its pooled and within-file calibration differ: generic Noul pooled ECE is 0.035, while median per-file ECE is 0.150. Context fields that define the label matter substantially. RLCD names Jev's own training, not an experiment training our generator with its rewards.

**Verified repository documentation and runner:** [RLCDAlignBench at `473b72f`](https://github.com/sumleo/RLCDAlignBench/tree/473b72f290234a49484c4d89c03aababcd74f758). The README provides offline metric reproduction from cached answers downloaded from its Hugging Face dataset. It distinguishes MIT code from CC BY-NC 4.0 released annotations/outputs and inherited upstream restrictions. This is useful reproducibility machinery, but its labels should not be substituted for our independent final-task verifier. The runner was inspected as an API/data-layout reference; no calls were executed in this audit.

### Jev-Mem: a third direct Jev paper missed previously

**Verified paper:** [arXiv:2609.23986v1](https://arxiv.org/html/2609.23986v1), submitted September 21, 2026. Jev-Mem uses a typed System-One controller for memory writing and retrieval: typing, redundancy filtering, relations, query routing, budget allocation, graph traversal, candidate scoring, and stopping. It reserves generative reasoning for synthesis. Its primary reported quality endpoint is memory-question-answering evaluation, including LLM judging; this is not evidence of independently verified scientific discovery.

**Verified code:** [Jev-Mem at `81574eb`](https://github.com/libingzheren/Jev-Mem/tree/81574eb23f3fd8d1a6c4d54a1e7d6f2dd539e9bb), including [typed policies](https://github.com/libingzheren/Jev-Mem/blob/81574eb23f3fd8d1a6c4d54a1e7d6f2dd539e9bb/memory/jev_mem_policies.py) and [client](https://github.com/libingzheren/Jev-Mem/blob/81574eb23f3fd8d1a6c4d54a1e7d6f2dd539e9bb/memory/jev_client.py). The policy layer batches relation questions and applies configured probability thresholds and graph-budget rules. Thus “first Jev supervisor,” “Jev controls depth/stopping,” or “typed fast semantic control” would be inaccurate broad claims.

## Named control and verifier papers: paper versus implementation

### CoT2-Meta

**Verified paper:** [arXiv:2603.28135v1](https://arxiv.org/html/2603.28135v1), submitted March 30, 2026. It explicitly combines structured process signals, a frontier, UCB-style selection, expansion, pruning, repair, stopping, and abstention/fallback under a shared budget. The practical controller is training-free. Appendix G.2 defines false-prune measurements using relaxed-pruning shadow search; Appendix I gives prompts and pseudocode.

It blocks a broad tree-management or false-pruning-audit novelty claim. I verified its methods and appendix descriptions, not its reported cross-benchmark improvements. Appendix I.6 describes reproducibility artifacts, but I did not verify a public official repository. Treat a local baseline as a reimplementation, and do not quote paper performance as locally reproduced. A statistically randomized, opportunity-cost-adjusted audit could be an extension, but ordinary shadow search is already present.

### GUARD

**Verified paper:** [Dissecting Failure Dynamics in Large Language Model Reasoning, arXiv:2604.14528v1](https://arxiv.org/html/2604.14528v1), submitted April 16, 2026. GUARD is the method name, not the paper title. It uses history-relative entropy spikes near segment boundaries, short local alternatives, entropy-based continuation selection, and late-stage control. Its recoverability analysis already samples alternative continuations from the same intermediate prefix. The paper supplies an official repository link.

**Verified implementation:** [math runner at `8d2bc7a`](https://github.com/ZHUWEI-hub/GUARD/blob/8d2bc7afcb3d070543604c4cf7830156b30aa486/eval/math_eval_guard.py). The checked defaults include entropy quantile 0.90 and three 100-token branches. Branch suffixes and temperatures differ; the score is entropy reduction plus a bonus for a detected boxed-answer marker. The cap loop advances using selected-branch tokens, while separate metrics record all generated branches. Therefore port its *policy* into our unified all-work ledger rather than assume its cap equals ours. Its boxed-marker detector should be audited before reuse. `ground_truth` is stored in objects and an answer-checking helper exists, but the inspected entropy-selection path does not call that helper; I found no basis to allege label leakage from this path alone.

### SAT

**Verified paper:** [arXiv:2604.07922v1](https://arxiv.org/html/2604.07922v1), submitted April 9, 2026. SAT steers Slow/Normal/Fast/Skip modes with a lightweight Pilot. Its Pilot fuses frozen GTE-small semantic embeddings and logit statistics through a small GRU, distilled from a teacher PRM. Its semantics-plus-uncertainty ablation makes a length/entropy-only baseline inadequate for our project. Skip changes subsequent generation; it does not establish that deleting an existing context segment is harmless.

**Verified implementation:** [Qwen3 runner at `88a6c3e`](https://github.com/byxw13/SAT_Code/blob/88a6c3e864948099afe3097c4fa8c2c33eb60224/run_think_control_V5_qwen3.py). `StepSeqPRM_GRU` concatenates logit and text features, projects them, runs a unidirectional GRU, and emits a step logit. This is a credible local feature/controller comparator. A portable reproduction requires obtaining its expected Pilot checkpoint and encoder assets, adapting machine-specific paths, and verifying decoding tags for the chosen backbone. I did not execute the runner or verify every training asset.

### Adaptive Test-Time Compute Allocation / AdaCompute

**Verified paper:** [arXiv:2604.14853v1](https://arxiv.org/html/2604.14853v1), submitted April 16, 2026. It solves per-input accuracy-versus-compute allocation with a Lagrangian and trains a cheap classifier to imitate oracle budget actions. It is mainly input-level allocation rather than repeated trace intervention. Learning a controller from offline measured outcomes is therefore not new even apart from the closer papers above.

**Code availability correction:** the paper advertises `zhiyuanZhai20/AdaCompute-LLM`, but both the GitHub page and API returned **404 on this audit date**. This establishes unavailable public access, not whether it is private, renamed, or removed. Do not list it as a verified runnable dependency.

### AVA

**Verified repository:** [AVA at `c277ad8`](https://github.com/llmsresearch/AVA/tree/c277ad827bdc35efef4063617f7c364c2d46359c). Its README links an [OpenReview paper](https://openreview.net/forum?id=JMDCMf7mlF) and claims TMLR 2026 publication. The OpenReview page presented a browser verification challenge in this audit, so venue status was not independently confirmed.

**Verified code:** [controller](https://github.com/llmsresearch/AVA/blob/c277ad827bdc35efef4063617f7c364c2d46359c/ava/controllers/ava_controller.py) allocates samples, depth, breadth, verification level, and tools through confidence-gap and budget thresholds. [Adaptive search](https://github.com/llmsresearch/AVA/blob/c277ad827bdc35efef4063617f7c364c2d46359c/ava/search/adaptive.py) computes its information-value score as uncertainty divided by `1 + 0.2 * depth`. It uses a deque with sorted sibling insertions, not a global learned action-value priority queue. Its final emergency generation path does not visibly debit the budget. These are specific source observations; they do not prove all AVA experiments used this module. Useful as a heuristic baseline, unsuitable as evidence that the field lacks learned control.

### RSD

**Verified paper:** [Reward-Guided Speculative Decoding for Efficient LLM Reasoning, arXiv:2501.19324](https://arxiv.org/abs/2501.19324). **Verified code:** [`main_online.py` at `c9a37c2`](https://github.com/BaohaoLiao/RSD/blob/c9a37c28e25306a7cfe9f8f3e1da5fdbc172f818/main_online.py), `get_responses`, lines 156–273. A draft step extends the prefix; the last PRM score is compared with a fixed threshold (default 0.7). Failed drafts are replaced by a target-model step, which enters the accepted collection without the same PRM gate being reapplied. The routine records discarded draft tokens.

This is a strong baseline for step acceptance/model routing, and a clean skeleton for adapter design. It does not directly compare the downstream values of repair, continuation, and branching. With a single generator and no stronger replacement model, calling an adapted threshold repair rule “RSD reproduction” would be misleading; name it “RSD-inspired gate.”

### GSI

**Verified current paper:** [Guided Speculative Inference for Efficient Test-Time Alignment of LLMs, arXiv:2506.04118v3](https://arxiv.org/abs/2506.04118v3), revised April 27, 2026. GSI combines reward-tilted soft best-of-N with speculative sampling. Its theoretical target is a reward-tilted distribution and its expected reward, not correctness under an arbitrary imperfect judge.

**Verified code:** [repository at `be4d17c`](https://github.com/j-geuter/GSI/tree/be4d17cda95036d235df7c0c9549e66b735e52a2), especially [`inference/speculative.py`](https://github.com/j-geuter/GSI/blob/be4d17cda95036d235df7c0c9549e66b735e52a2/inference/speculative.py). It contains both tilted decoding and threshold variants, using reward plus big/small model log probabilities. **The README explicitly says the released implementation corresponds to an older paper version.** Its example setup uses separate small, large, and reward-model workers. It is relevant if we study multi-model routing; it is not the cheapest first comparator for a single-generator Mac pilot.

### PAV / Rewarding Progress

**Verified paper and methods:** [arXiv:2410.08146v1](https://arxiv.org/html/2410.08146v1). It defines process advantages under a prover policy and combines them with outcome information for search and RL. Its labels use sampled continuations of prefixes. The prover's complementarity to the base policy matters; a stronger prover is not automatically a better source of dense progress rewards.

It blocks “progress rather than local validity” as a new idea. Our meta-actions differ from its object-level next reasoning steps, but that distinction alone is no longer sufficient given Calibration Is Not Control and DIAL. An official repository was not verified. A search result titled an implementation of this paper explicitly identifies itself as a third-party attempt; do not mistake it for the authors' code.

### ThinkPRM

**Verified paper:** [Process Reward Models That Think, arXiv:2504.16828v5](https://arxiv.org/abs/2504.16828v5), revised December 8, 2025. It trains a generative step verifier, supports verification compute scaling, and evaluates best-of-N and guided search. It provides a local reasoning-verifier comparison, not ground truth.

**Verified implementation:** [`beam_search.py` at `045b1ae`](https://github.com/mukhal/ThinkPRM/blob/045b1aed5d72301690a8fc08e846c90b757672c8/search-and-learn/src/sal/search/beam_search.py). It exposes `prm.score`, aggregates step scores, filters exact duplicate beam text, tracks completion, and retains a configured number by ranking. It rebuilds chat-template prompts and uses fixed iterations/beam settings; exact-token replay and our hard shared budget require additional instrumentation. The file carries an Apache-2.0 notice although the repository-level license is MIT: preserve file-level licensing when reusing it.

### Dyve

**Verified paper:** [arXiv:2502.11157v1](https://arxiv.org/abs/2502.11157v1), February 16, 2025. It combines rapid judgments with longer verification and curates supervision with Monte Carlo estimates and LLM evaluation. Adaptive verifier effort is prior art.

**Verified repository documentation:** [Dyve at `d3697a2`](https://github.com/staymylove/Dyve/tree/d3697a2d2b64e34c4ab460181dc67b03957f036c). The README links a 14B model and a CoT dataset. Those links and setup instructions were inspected; the entire inference/training implementation was not audited or run. Its 14B verifier is less practical than a small local judge for the first constrained-hardware pilot.

### Steer, Don't Solve

**Verified paper:** [arXiv:2606.21811v2](https://arxiv.org/abs/2606.21811v2), revised September 1, 2026. **Verified code/docs:** [critic-training at `1c4aa0d`](https://github.com/shubhamrgandhi/critic-training/tree/1c4aa0d3c28c522fe9e7541b14b86fe0c4a13e5c), especially [`default_prm.py`](https://github.com/shubhamrgandhi/critic-training/blob/1c4aa0d3c28c522fe9e7541b14b86fe0c4a13e5c/mini-swe-agent/src/minisweagent/agents/default_prm.py). It calls a structured critic every configurable number of agent steps and includes prefix replay. The README describes SFT from teacher critiques and DPO from ten candidate critiques ranked best/worst by a judge.

The documented preference supervision differs from executing each critique and comparing its downstream success. That distinction remains useful experimentally, but it is not an unexplored intervention-learning concept overall. Use this codebase for a later code-agent extension, not as a necessary dependency for the first mathematical/procedural pilot.

## Training and exploration boundaries

| Source | Verified scope and consequence |
|---|---|
| [PURE, arXiv:2504.15275v3](https://arxiv.org/html/2504.15275v3) | Min-form process credit assignment alleviates the failures of summing PRM rewards. The paper also reports later reward hacking under PRM-only training; min-form is not a proof that hacking is solved. The [`verl` README at `b174e50`](https://github.com/CJReinforce/PURE/blob/b174e508a4469663543cfcb366be78cf199d20bf/README.md) documents PRM-only, verified-only, and mixed modes. Defer generator RL until inference-control value is established. |
| [Monitoring Reasoning Models…, arXiv:2503.11926v1](https://arxiv.org/html/2503.11926v1) | In the studied coding/RL setting, low optimization pressure can help; stronger direct monitor optimization induces less legible reward hacking. It supports independent behavioral evaluation and held-out audits, not the claim that all process supervision necessarily causes obfuscation. No local reproduction performed. |
| [rStar-Math, arXiv:2501.04519](https://arxiv.org/abs/2501.04519) | Search, process evaluation, and iterative policy improvement are established. The requested [`rStar-math` branch](https://github.com/microsoft/rStar/tree/b3c1846185f4af877e1e441a67c4c5fcfed9d442) resolves to `b3c1846`; current default `main` is not a safe substitute. Branch existence and paper were verified; source-level reproduction was not. |
| [Amortizing Intractable Inference…, arXiv:2310.04363](https://arxiv.org/abs/2310.04363) | GFlowNet-based amortized inference gives relevant diversity-seeking context. Paper existence/abstract verified; detailed code not audited here. It does not establish Jev-mediated scientific novelty. |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6), [AlphaEvolve, arXiv:2506.13131](https://arxiv.org/abs/2506.13131) | Their discovery framing uses evaluated candidate artifacts. Primary sources verified at overview level. They motivate executable final evaluation if we later move to algorithm discovery; they do not supply evidence that judge-perceived novelty is a valid innovation endpoint. |

## Selective acquisition is also an existing research area

[RACER, arXiv:2605.10805v1](https://arxiv.org/html/2605.10805v1) routes between reasoning and non-reasoning judges under a cost constraint and distributional robustness formulation. [LLM-as-Judge on a Budget, arXiv:2602.15481](https://arxiv.org/abs/2602.15481) allocates repeated judgments according to estimated variance. [Instance-Optimal Estimation with Multiple LLM Judges on a Budget, arXiv:2605.23362](https://arxiv.org/abs/2605.23362) studies cost-aware multi-judge score estimation. Their targets are judgment quality/estimation, which differs from *change in verified task outcome caused by the selected reasoning action*, but “adaptive allocation of judge calls” is not itself new.

For a stable decision-only service, repeating the identical question may produce no new information. Spend the initial Jev allocation on distinct natural states and matched outcomes. Acquire additional rubric components only if a development experiment shows they change useful action decisions. A latency-aware acquisition policy must be compared with always-query and a matched query-rate random policy, not just no-query.

## Defensible differentiators to test

These are proposed research hypotheses, not verified gaps or promised results:

1. **Representation transfer at a measured adaptation budget.** Freeze a small task-agnostic typed schema; test whether it retains action-value prediction and policy value across a new generator/horizon/task family with zero or few new continuation labels. Compare against frozen embeddings, cheap features, and destination-trained versions. Report negative transfer. DIAL's per-setting learning is the relevant baseline; existing cross-model analyses mean that merely reporting transfer is insufficient.
2. **Acquisition value measured by task outcome.** Learn whether paying for Jev changes the *best intervention* enough to improve net utility, rather than whether it reduces judgment error. Keep acquisition and intervention costs inside the same state and budget. This could extend judge routing, but requires broader active-feature-acquisition and metareasoning positioning before a novelty claim.
3. **Sequential deployment after repeated prefix experiments.** First reproduce the offline control result with independent continuation replications; then deploy the frozen controller at several checkpoints and quantify the difference between offline value estimates and realized outcomes. Study controller-induced state shift directly. Offline prefix success cannot certify sequential safety.
4. **Exploration-role preservation.** Predefine tests that distinguish a conjecture, a failed but informative experiment, a retraction, and an erroneous assertion. Measure useful-branch loss through actual continuations. Do not label every uncertain statement as valuable, or claim novelty simply from using semantic role labels.

The smallest informative contribution is a reproducible benchmark of **when semantic judgment is worth purchasing for reasoning control**, even if Jev is no better than a local representation. If improvements disappear after feature/cost matching, the correct conclusion is a negative result for the Jev-centered hypothesis.

## Baseline priority and budget-conscious execution

| Priority | Comparator | Purpose |
|---|---|---|
| Required | Uninterrupted generator; sham pause/resume | Establish the actual intervention effect and exclude resume artifacts. |
| Required | Always continue, always repair, always branch; rate-matched random actions | Show that state-dependent decisions improve over static treatment and trigger frequency. |
| Required | Per-action learned controller with strong cheap/lexical/embedding features | Direct comparison for the incremental Jev contribution. |
| Required | Same controller and schema with a local open-weight typed judge | Test whether hosted Jev offers a useful quality/cost tradeoff. |
| Required | DIAL-style sparse learned gate; conservative per-action random forest inspired by Calibration Is Not Control | Compare against the closest learned-control prior art, clearly labeled reimplementations where code is unverified. |
| First published inference baseline | GUARD-style local branching/entropy gate with our shared ledger | Closely aligned with one-trace interventions and does not need an extra large model. |
| If depth/length is claimed | SAT-style embedding-plus-logit Pilot/FSM | Strong inexpensive semantic and runtime control baseline. |
| If a frontier is claimed | ThinkPRM-guided beam search; CoT2-Meta policy reimplementation | Separates evaluator changes from tree management. |
| If multi-model routing is claimed | RSD; GSI with paper-version caveat | Necessary when the proposed advantage is accepting/replacing steps across model sizes. |
| If code-agent guidance is claimed | Fixed-interval small critic | Distinguishes intervention timing from stronger generated advice. |

Do not attempt every method in the first pilot. For the three-action frozen-generator design, the required rows plus one GUARD-style comparator are sufficient to test headroom and attribution. Add named systems only when their mechanism matches the final claim. Hardware/backend adaptations must preserve the observable features or disclose their absence; an API-only baseline cannot fairly be penalized for lacking full-logit entropy.

## Reuse and evidence limitations

GitHub metadata reported clear root licenses for RSD (Apache-2.0), GSI (Apache-2.0), AVA (MIT), ThinkPRM (MIT), Jev-Mem (MIT), RLCDAlignBench (MIT), and rStar (MIT). It did not identify a root license for SAT, GUARD, Dyve, critic-training, or PURE at the inspected heads. This is a *limited license-availability observation*, not a legal determination about every embedded component. Keep the first implementation independent; preserve verified file-level notices for any copied code.

All papers remain external evidence. We verified titles, dates, stated mechanisms, and selected code paths; none of their result tables became this project's experimental results. The previous chat's downloadable package, synthetic numbers, tests, and hardware estimates were not independently recreated by this literature audit. Paper performance claims should be attributed to their authors, and project reports should distinguish planned, simulated, and actually executed experiments.
