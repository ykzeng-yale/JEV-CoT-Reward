# Theory: outcome-grounded interventions in reasoning

Status: mathematical development and assumptions, 2026-09-27. This document contains no empirical Jev or LLM performance results. The results below specialize standard experimental identification, policy learning, and metareasoning arguments; they are not claimed as new general theorems.

## 1. Research object and notation

An intervention should be judged by its effect on an independently verified terminal outcome. A probability that a prefix is correct is a different estimand. Process advantage verification already makes the distinction between correctness and downstream progress. [Rewarding Progress](https://arxiv.org/abs/2410.08146)

**Novelty correction.** The central intervention-value framing and same-prefix experiments are already explicit in [Calibration Is Not Control](https://arxiv.org/html/2606.21399v1): it defines intervention advantage, studies control sufficiency, executes candidate actions from shared prefixes, and fits an action-conditioned controller. [DIAL](https://arxiv.org/html/2605.06908v1) also learns the direction of useful intervention using checkpoint exploration and counterfactual continuation. [AERA](https://arxiv.org/html/2608.27964v1) learns residual value of further inference. Consequently, this project's action-value theory and replay design are replication foundations. The narrower research hypothesis is whether Jev's typed semantics supply incremental, transferable **decision value after acquisition costs**, and whether selective acquisition improves that tradeoff. Neither claim is established by this document, and value-of-information gating is itself standard metareasoning.

Let a checkpoint be the full observable history

\[
H=(X,\tau,\mathcal F,E,B,G),
\]

where \(X\) is the task, \(\tau\) contains exact token IDs and formatting, \(\mathcal F\) is the candidate frontier, \(E\) is external/tool state, \(B\) is remaining resources, and \(G\) fixes generator weights, quantization, tokenizer, sampling, and backend. Previous expenditures are recorded. Conditioning on text alone need not identify the same state.

The initial action set is \(\mathcal A=\{c,r,b\}\): continue, local repair, and local branch. An action is a complete implementation contract, including prompts, segment limits, selector, and budget accounting. Let \(\mu\) be the *fixed subsequent continuation policy*. Define

\[
Q^\mu(h,a)=\mathbb E[U^{a,\mu}\mid H=h],\qquad
\Delta^\mu(h,a)=Q^\mu(h,a)-Q^\mu(h,c).
\]

\(Y^{a,\mu}\in\{0,1\}\) is strict terminal success. The implemented development experiment uses \(U=Y\) under an **emitted generator-token envelope**; judge acquisition, inserted prompt tokens, repeated prefill, and wall time are additionally recorded and are not equalized by that envelope. A future full-resource experiment must define its resource vector explicitly. Secondary analyses can use \(U=Y-\lambda C\), provided units and \(\lambda\) are fixed before evaluation. Bounded-concentration claims below explicitly use \(U\in[0,1]\); cost-penalized utility must be rescaled or its actual range substituted.

Potential outcomes are random because continuations are random. They do not assert that there is one uniquely correct counterfactual future. A cloned checkpoint permits repeated samples from intervention distributions; it does not reveal a trajectory's exact individual causal effect. Changing \(G\), \(B\), the selector, or \(\mu\) changes the estimand.

## 2. Identification by checkpoint replay

**Proposition 1.** Suppose (i) actions and the continuation policy are well defined; (ii) replay restores the same checkpoint information and environment; (iii) one replay does not affect another; (iv) the action assignment is independent of potential outcomes conditional on \(H\); (v) every action of interest has positive probability; and (vi) failures and truncations are retained according to a prespecified rule. Then

\[
Q^\mu(h,a)=\mathbb E[U\mid H=h,A=a].
\]

**Proof.** Consistency gives \(U=U^{a,\mu}\) on \(A=a\). Conditional randomization gives
\(\mathbb E[U^{a,\mu}\mid H=h,A=a]=\mathbb E[U^{a,\mu}\mid H=h]\). Positivity makes the first conditional mean observable. Combining these identities proves the result. \(\square\)

When all actions run at every checkpoint, there is no missing treatment arm to inverse-weight. For checkpoint \(h_i\),

\[
\widehat Q_i(a)=\frac1{m_{ia}}\sum_{r=1}^{m_{ia}}U_{iar}
\]

is conditionally unbiased when the positive replicate count \(m_{ia}\) is fixed independently of those replays' outcomes and each replay has the correct marginal continuation distribution. Outcome-dependent stopping generally invalidates this assertion for the stopped sample mean, even when every next draw is conditionally unbiased. A separate fixed validation batch, an appropriate sequential estimator, or a prespecified sequential analysis is needed after adaptive allocation. Independent seeds simplify variance analysis. Common random numbers may be used deliberately, but equal integer seeds after diverging token histories do not automatically create informative paired futures. Randomize run order to avoid confounding action with server drift or thermal throttling.

Faithful replay can recompute a KV cache from token IDs; it need not serialize the cache. Exact token IDs alone do not guarantee exact continuation behavior if hidden backend/sampler state changes. A sham pause/resume comparison tests this issue. Repair deliberately modifies a prefix; continue must not silently change the chat template or insert a new instruction.

**Constructed states, eligibility, and the target population.** Let \(R\) be a frozen prefix-construction procedure and \(E\) indicate checkpoint eligibility, determined before intervention assignment. The replay estimand is \(Q_R^\mu(h,a)\) on \(P_R(H\mid E=1)\). The original implementation generates a fixed block, excludes blocks that already completed or entered the answer phase, then rewinds to its last retained paragraph. This is a well-defined *generate-then-rewind* procedure, but it is not an online stopping time at the retained paragraph: eligibility and boundary choice can depend on discarded later tokens. Its cost counters are available at the actual post-generation decision. They would be future information for a different system making its decision at the earlier paragraph. Discarded-token content/statistics must not be smuggled into a claimed earlier checkpoint.

All-arm replay remains valid for the specified constructed-state distribution. To claim value for online paragraph-boundary interventions, either sample boundaries prospectively as generation streams or evaluate the same generate-then-rewind procedure as part of deployment. Exact prefix equality alone does not establish equality of these state distributions. An uninterrupted comparison must account for the prefix-construction procedure itself.

If two policies use the same \(R\), intervene at most once, and use identical fallback behavior when \(E=0\), their full-episode difference is

\[
V_R(\pi)-V_R(\pi_0)
=\Pr_R(E=1)\,\mathbb E[Q_R^\mu(H,\pi(H))-Q_R^\mu(H,\pi_0(H))\mid E=1].
\]

**Derivation.** Decompose the mean outcome difference by \(E\); the ineligible term is zero by the common-fallback assumption, and apply the replay conditional means to the eligible term. This identity fails when policies alter prefix construction, eligibility, later interventions, or ineligible fallback. With several checkpoints per problem, an unweighted checkpoint average also overweights problems producing more checkpoints; specify a within-problem sampling distribution and give each original problem its intended population weight. Problem weighting does not turn a single-intervention target into a sequential one.

**Sparse-arm extension.** For a frozen deterministic policy \(\pi\), known assignment probabilities \(e_a(h)>0\), and outcome functions \(m_a(h)\) fitted independently of the evaluation observation, define

\[
\phi_\pi=m_{\pi(H)}(H)
+\frac{\mathbf1\{A=\pi(H)\}}{e_A(H)}[U-m_A(H)].
\]

Conditioning on \(H\) and fitted functions, the second term has expectation \(Q^\mu(H,\pi(H))-m_{\pi(H)}(H)\). Therefore \(\mathbb E\phi_\pi=\mathbb E Q^\mu(H,\pi(H))\). If estimated propensities are used, double robustness requires its own consistency conditions; calling an arbitrary weighted estimator “DR” does not supply them. Cross-fitting must split by original problem. This is standard contextual-bandit evaluation. [Dudík, Langford, and Li](https://arxiv.org/abs/1103.4601)

## 3. What semantic information can and cannot add

Let \(W\) denote inexpensive features and \(J\) denote semantic judge features. For this section, acquisition does not alter outcomes, budgets, or the action set. Set

\[
V^*_{W,J}=\mathbb E\max_a\mathbb E[U^a\mid W,J],\qquad
V^*_W=\mathbb E\max_a\mathbb E[U^a\mid W].
\]

**Proposition 2.** \(V^*_{W,J}\ge V^*_W\).

**Proof.** A policy measurable with respect to \(W\) remains admissible when \((W,J)\) is observed: it can ignore \(J\). Taking the supremum over the larger policy class cannot decrease value. Equivalently, conditional Jensen's inequality for the maximum gives the result. \(\square\)

Strict improvement requires decision-relevant heterogeneity: after conditioning on \(W\), the extra signal must sometimes distinguish situations with different favorable actions. Accurate semantic classification can have zero decision value if continue is optimal everywhere. A finite fitted learner can also perform worse because it has estimation error.

If \(J=f(H)\), or its randomization is independent of rollout outcomes conditional on \(H\), Jev supplies no additional external information beyond full \(H\). The scientific hypothesis is that it provides a useful, inexpensive representation compared with feasible alternatives, not that it creates information absent from the trace.

**Cost correction.** Let \(B\) be a resource vector and \(c_J\) the acquisition cost. The operational pre-query comparison is

\[
V_{\mathrm{query}}(w,B)=
\mathbb E_{J\mid w,B}\left[\max_{a\in\mathcal A(B-c_J)} Q(w,J,B-c_J,a)\right]
\quad\text{versus}\quad
V_{\mathrm{noquery}}(w,B)=\max_{a\in\mathcal A(B)} Q_0(w,B,a).
\]

This display assumes deterministic known acquisition cost conditional on the pre-query state, available acquisition resources, and that acquisition changes subsequent transitions only through its reported information and remaining budget. For stochastic latency, failure, or usage, integrate over the complete acquisition outcome \(O=(J,C_J,\text{status})\); define its fallback and feasible action set, including exhaustion, inside that expectation. If acquisition changes external tool/world state, include the updated state as well. A vector subtraction cannot represent these effects without a transition model.

Use cost penalties and/or hard constraints only as explicitly declared in the objective; do not subtract a fee twice inadvertently. A hard budget plus a separate cost preference can be intentional, but is a different objective. The expected maximum must occur *inside* the expectation over a not-yet-observed signal. Comparing two selected actions after the query ignores the decision to buy the query. Runtime, latency, local inference, and hosted API spend are distinct resources, not interchangeable tokens. This is a metareasoning problem. [Selecting Computations](https://arxiv.org/abs/1207.5879)

## 4. Prediction error and decision regret

**Proposition 3.** Let \(a^*(h)\in\arg\max_a Q^\mu(h,a)\) and \(\widehat a(h)\in\arg\max_a\widehat Q(h,a)\). If, at \(h\),

\[
\max_a|\widehat Q(h,a)-Q^\mu(h,a)|\le\epsilon(h),
\]

then \(Q^\mu(h,a^*)-Q^\mu(h,\widehat a)\le2\epsilon(h)\).

**Proof.** Add and subtract \(\widehat Q(h,a^*)\) and \(\widehat Q(h,\widehat a)\). The difference between these two fitted values is nonpositive by greedy selection; the other two terms are at most \(\epsilon(h)\) each. \(\square\)

The expected regret is consequently at most \(2\mathbb E\epsilon(H)\) if this statewise bound is valid. If a unique optimal action has gap \(g(h)>2\epsilon(h)\) to every alternative, the selected action must be optimal. With a uniform bound \(\epsilon\), regret is at most \(2\epsilon\Pr\{g(H)\le2\epsilon\}\), apart from arbitrary tie choices with zero regret.

Representation and fitting errors can be separated by setting \(q_a(z)=\mathbb E[Q^\mu(H,a)\mid Z=z]\). If \(|Q^\mu(h,a)-q_a(z(h))|\le\epsilon_{\rm repr}\) and \(|q_a(z(h))-\widehat q_a(z(h))|\le\epsilon_{\rm est}\), then the proposition applies with their sum. Such uniform bounds are strong assumptions, not consequences of validation-set AUROC or calibration.

**Finite replay precision.** For \(M\) fixed checkpoints, \(K\) actions, and \(m\) independent Bernoulli replays per pair, Hoeffding and a union bound give

\[
\Pr\left\{\max_{i,a}|\widehat Q_i(a)-Q_i(a)|>\epsilon\right\}
\le2MK\exp(-2m\epsilon^2).
\]

Thus \(m\ge\log(2MK/\delta)/(2\epsilon^2)\) suffices. With \(M=48,K=3,\delta=.05,\epsilon=.10\), the bound requires 433 replays per action-checkpoint pair. Four replays are useful for pooled modeling and screening; they do not justify confident individual best-action labels. Even 12–16 replays provide limited precision. Fit Bernoulli outcomes, retain their replication counts, and evaluate selected policies on independent problems or fresh rollouts.

## 5. Sequential improvement: valid theorem, demanding premises

Use a finite-horizon history-state decision process. Include budget, step count, tool state, and judge-acquisition phase in \(H_t\); absorb early termination with zero future rewards. Let

\[
Q_t^\mu(h,a)=\mathbb E[r_t+V_{t+1}^\mu(H_{t+1})\mid h,a],\quad
A_t^\mu(h,a)=Q_t^\mu(h,a)-V_t^\mu(h).
\]

Assume the baseline \(\mu\) chooses deterministic \(a_{0t}(h)\). Both policies have the same initial distribution, transition law, reward definition, and resource envelope.

**Proposition 4.** If, on every history reachable by deployed policy \(\pi\), all action errors obey \(|\widehat Q_t-Q_t^\mu|\le\epsilon_t(h)\), and \(\pi\) deviates from \(a_{0t}\) only when

\[
\widehat Q_t(h,a)-\widehat Q_t(h,a_{0t})>2\epsilon_t(h),
\]

then \(V^\pi\ge V^\mu\).

**Proof.** At a switched state, the true difference \(Q_t^\mu(h,a)-Q_t^\mu(h,a_{0t})\) is at least the fitted difference minus \(2\epsilon_t(h)\), hence positive. At an unswitched state it is zero. Since \(V_t^\mu(h)=Q_t^\mu(h,a_{0t})\), the true advantage is nonnegative everywhere \(\pi\) visits. Under \(\pi\),

\[
\mathbb E\sum_{t=0}^{T-1}A_t^\mu(H_t,A_t)
=\mathbb E\sum_t[r_t+V_{t+1}^\mu(H_{t+1})-V_t^\mu(H_t)]
=V^\pi-V^\mu,
\]

where the first equality uses conditional expectation, and the second telescopes with zero terminal value. Its left side is nonnegative. \(\square\)

This finite-horizon identity is a standard performance-difference argument. [Kakade and Langford](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)

Three restrictions matter in this project:

1. Calibration on naturally occurring baseline prefixes gives no uniform bound on the new histories reached after repairs and branches. Sequential deployment must be tested directly.
2. \(Q^\mu\) uses baseline continuation after *each* candidate action. Greedy improvement remains valid under repeated deployment only because the pointwise premise covers all resulting histories.
3. A gate applied only **after paying for Jev** can justify improvement over another policy that has already paid. It cannot justify improvement over a policy that never queries Jev. The pre-query acquisition action must separately have nonnegative advantage, or the entire query-plus-intervention must be evaluated as one cost-inclusive action.

Jev confidence is not \(\epsilon_t(h)\). A fitted confidence interval is not automatically simultaneous over adaptively selected histories. If the bound fails on an event of probability \(\delta\) over fitting, the theorem is only a high-probability claim on the complement; do not convert it into an unconditional safety guarantee.

## 6. Honest policy evaluation and conservative certificates

**Proposition 5.** Freeze \(K\) candidate policies before evaluating independent problems. For policy \(k\), let \(D_{ik}\in[-1,1]\) be the per-problem difference in mean success over a fixed number of seeds, relative to a fixed baseline. Then, with probability at least \(1-\alpha\), simultaneously for all \(k\),

\[
\Delta_k\ge\overline D_k-\sqrt{\frac{2\log(K/\alpha)}n}.
\]

**Proof.** Hoeffding for independent observations with range width two gives
\(\Pr\{\overline D_k-\Delta_k\ge t\}\le\exp(-nt^2/2)\). Set \(t=\sqrt{2\log(K/\alpha)/n}\) and union-bound over \(K\) policies. Correlation between policies or seeds *within* a problem does not violate independence across problems. \(\square\)

At \(n=3000,K=1,\alpha=.05\), the radius is 0.04469. At \(n=48\), it is about 0.3533. The mechanism screen cannot certify a small improvement by this worst-case bound. An ordinary paired interval can be much narrower, but has different assumptions and is not a distribution-free certificate. Conditional on the separate training data, a frozen learned policy meets the “frozen” requirement; a policy fitted on the evaluation problems does not.

If strata have fixed unequal target weights, apply a weighted bound or separate bounds; do not substitute the unweighted formula. For problem weights \(w_i\ge0,\sum_iw_i=1\), the analogous radius is \(\sqrt{2\log(K/\alpha)\sum_iw_i^2}\). Proof: apply weighted Hoeffding with widths \(2w_i\). Adaptive repeated peeking needs a prespecified sequential method or alpha spending; an ordinary final-sample interval does not authorize stopping whenever it becomes significant.

Policy optimization and evaluation separation is fundamental in policy learning. [Athey and Wager](https://arxiv.org/abs/1702.02896)

**Cross-fitting is not a frozen-policy test.** In \(F\)-fold cross-fitting, prediction for problem \(i\) excludes its outcomes, which prevents its direct training leakage. However, fitted policies for different folds share training problems. The out-of-fold contributions are therefore generally dependent: problem \(i\)'s outcomes may affect the policy applied to problem \(j\), and conversely. Resampling only the resulting contributions without refitting has no generic nominal coverage guarantee and Proposition 5 does not apply. This is more than a missing “training variance” component. An exploratory cross-fit mean describes a collection of fold-trained procedures, not automatically the final policy refitted on all observations. Refitting the complete pipeline in a nested cluster bootstrap can study instability, but is not universally valid for discontinuous policy selection; use a disjoint test set for a straightforward confirmatory claim.

Further requirements for inference are operational: the original problem (or broader shared generator/template unit, if coupled) is the independent unit; all candidate models, prompts, selection rules, and feature schemas chosen using a test set count as test reuse; and a global query quota that binds adaptively across test problems creates dependence unless that resource allocation is part of the tested sequential protocol. Separate per-problem quotas, or a nonbinding prespecified study cap, avoid that particular coupling. Repeat seeds reduce Monte Carlo noise but do not increase the number of independent problems. Stratified bootstrap estimates the declared stratum-weighted population, not arbitrary new task families.

For a frozen policy on a finite list of observed checkpoints, fresh replay seeds support inference about continuation randomness **on that list**. Generalization to new problems requires the problem-sampling assumption. Neither a fixed-list replay interval nor a tiny-sample problem bootstrap establishes broad task-family generalization.

## 7. Pruning audits estimate a precise replay target

Fix a set of \(N\) logged states where a frozen controller proposed pruning. For state \(i\), define the matched-budget replay difference

\[
L_i=Q(h_i,\text{keep and reallocate})-Q(h_i,\text{prune and reallocate}).
\]

The entire subsequent frontier scheduler and opportunity cost are included in the two replay protocols. Draw audit indicator \(R_i\) with known probability \(p_i>0\), independent of future replay outcomes conditional on the recorded state. Let \(\widehat L_i\) be an unbiased replay estimate.

**Proposition 6.** \(\widehat L=N^{-1}\sum_iR_i\widehat L_i/p_i\) is unbiased for \(N^{-1}\sum_iL_i\).

**Proof.** Conditional on the logged states, independence and unbiasedness give \(\mathbb E[R_i\widehat L_i/p_i]=p_iL_i/p_i=L_i\). Sum and divide by \(N\). \(\square\)

Small audit probabilities can cause severe variance. The target is expected loss on the logged proposed-pruning distribution, not the effect of an updated controller on a new deployment distribution. Updating the controller with audits then evaluating on those same replays is adaptive reuse. A branch's success under extra, uncharged compute does not estimate the cost of pruning under a fixed global budget. This extension is deferred until the three-action study succeeds.

## 8. Transfer conditions and their limitations

Let \(P_S,P_T\) be source and target checkpoint distributions over an aligned state space, with the same actions and utility scale. Define \(r_D(h,\pi)=\max_aQ_D(h,a)-Q_D(h,\pi(h))\). Assume

\[
P_T\ll P_S,\qquad \frac{dP_T}{dP_S}\le\kappa,\qquad
\sup_{h,a}|Q_T(h,a)-Q_S(h,a)|\le\zeta.
\]

**Proposition 7.** \(\mathrm{Regret}_T(\pi)\le\kappa\mathrm{Regret}_S(\pi)+2\zeta\).

**Proof.** At every \(h\), \(\max_aQ_T(h,a)\le\max_aQ_S(h,a)+\zeta\) and \(Q_T(h,\pi(h))\ge Q_S(h,\pi(h))-\zeta\). Hence \(r_T\le r_S+2\zeta\). Integrating under \(P_T\) and using nonnegativity of \(r_S\) with the density-ratio bound gives the conclusion. \(\square\)

New task families may violate overlap. A new generator, quantization, context format, or budget can substantially change \(Q\); the bound may then be uninformative. Exact full-history distributions are usually poorly overlapping. An aligned low-dimensional representation may make an empirical comparison possible, but introduces representation error and does not prove these assumptions. No arbitrary-OOD, innovation, or alignment guarantee follows.

## 9. Selective feature acquisition as a measurable decision

Fix a cheap-feature policy \(\pi_0\) and a policy \(\pi_J\) that obtains Jev features before selecting an intervention. Define two *complete* potential protocols at the pre-query checkpoint: \(U_0\) executes \(\pi_0\); \(U_J\) purchases the query, accounts for latency and resources, then executes \(\pi_J\). The task, budget, and all downstream rules are fixed. Let

\[
d(w)=\mathbb E[U_J-U_0\mid W=w],\qquad
\eta(w)\in\{0,1\}
\]

be acquisition advantage and a gate that cannot inspect \(J\) before deciding whether to buy it.

**Proposition 8.** The value of gate \(\eta\) is

\[
V(\eta)=\mathbb E U_0+\mathbb E[\eta(W)d(W)].
\]

The optimal gate among measurable functions of \(W\) is \(\eta^*(w)=\mathbf1\{d(w)>0\}\), with arbitrary tie behavior. Its regret is

\[
V(\eta^*)-V(\eta)=
\mathbb E[|d(W)|\mathbf1\{\eta(W)\ne\eta^*(W)\}].
\]

If \(|\widehat d(w)-d(w)|\le\epsilon\) and \(\widehat\eta(w)=\mathbf1\{\widehat d(w)>0\}\), then

\[
V(\eta^*)-V(\widehat\eta)
\le\epsilon\Pr\{|d(W)|\le\epsilon\}.
\]

**Proof.** The gate executes \(\eta(W)U_J+(1-\eta(W))U_0\); condition on \(W\) to obtain the value identity. Maximizing the integrand gives the oracle sign rule. At a wrong sign, the lost value is \(|d(w)|\), which gives the regret identity. A wrong estimated sign implies \(|d(w)|\le|\widehat d(w)-d(w)|\le\epsilon\); integrate to obtain the final bound. \(\square\)

This is a standard binary treatment/measurement-selection argument, not a new general theorem. It suggests a distinct measurable target: not “is Jev uncertain?” or “do the judges disagree?”, but “does acquiring Jev improve the implemented policy enough here to cover its cost?” Fit and freeze \(\pi_0,\pi_J\) before using held-out replays to label acquisition value, or use nested problem-level cross-fitting. Reusing the same noisy outcomes to optimize both policies and fit/evaluate the gate gives optimistic results. A comparison that assumes already-paid Jev responses estimates information value, not acquisition value.

Conditional purchase may be unnecessary if always-Jev is effectively free at the measured workload, or unhelpful if Jev has no value anywhere. Local-judge purchase is an equivalent comparator. Deadline-constrained experiments need actual replay under each acquisition protocol; subtracting average latency from a token-matched result does not generally reconstruct its outcome.

## 10. What would constitute new evidence

The [binary-outcome baseline appendix](baseline_contracts.md) derives an additional decision-loss identity: outright-win probability and expected intervention gain differ when outcomes tie. It gives the discordance-conditioned gate target, equal-problem weighting rule, and a population logistic-loss-to-value bound. These are standard binary-learning arguments specialized to the actual endpoint, not evidence that a fitted gate or Jev improves outcomes.

The mathematical results establish what a properly executed experiment measures and when particular implications are valid. Because prior work already targets intervention effects, the Jev-specific empirical contribution must establish that typed semantic features improve *held-out policy value* over strong cheap and local-judge features; gains survive feature-acquisition costs; and selective acquisition, repeated deployment, or a specified transfer test adds useful evidence beyond the existing intervention-control literature. A study can also deliver a rigorous negative result rather than a new controller.

A negative Jev result remains informative. Improved offline feature prediction without improved deployment utility is not a successful control method. An observed controller improvement with no incremental Jev benefit supports learned metareasoning, not a Jev-specific claim.

For concrete falsification, freeze the generator \(G\), prefix builder \(R\), continuation policy \(\mu\), selector, budget, learner, and training/tuning allotment. On an independent test set, compare fitted policies \(\pi_W\), \(\pi_{W,J}\), and \(\pi_{W,L}\), where \(L\) is the independently prompted local judge. The replay information contrast is

\[
\theta_{J:W}^{\rm replay}
=\mathbb E_{H\sim P_R(\cdot\mid E=1)}
[Q_R^\mu(H,\pi_{W,J}(H))-Q_R^\mu(H,\pi_W(H))].
\]

It is estimated without an oracle by averaging the test replays for each policy's selected action, with paired original-problem weighting. Its local-judge analog substitutes \(\pi_{W,L}\) for \(\pi_W\). These contrasts measure the **implemented frozen learners**; they do not identify the Bayes-optimal information value in Proposition 2, and finite-data failure does not prove that Jev contains no useful information.

The purchase contrast \(\theta_{J:W}^{\rm deploy}=\mathbb E[U_J-U_W]\) instead executes the complete cost-inclusive protocols from task start, including acquisition failures and ineligible fallback. A positive replay contrast can coexist with a negative deployment contrast. For query-rate-matched controls, select the random-acquisition probability on development data and then freeze it; matching the rate retrospectively using test Jev outcomes changes the tested procedure. Report a prespecified practical margin and intervals for these contrasts. “No statistically significant benefit” is not an equivalence result; exclude the practical margin with a justified upper interval before claiming that a sufficiently large benefit is absent in the tested setting.

## 11. Executed finite validation

The repository's [validation script](../scripts/validate_theory.py) was run with seed 1729 and its [manifest](../results/theory_validation.json) records fourteen passing checks. Eleven tests in [test_theory.py](../tests/test_theory.py) passed. These are finite synthetic checks, not proofs of general validity and not Jev/LLM experiment results.

| Check | Observed result |
|---|---|
| Performance-difference identity in 200 four-stage, three-state, three-action MDPs | Largest absolute discrepancy \(1.31\times10^{-16}\) |
| Conservative gate with verified uniform fitted-value error in those MDPs | No negative improvement; 233 nonbaseline state-action choices |
| Greedy regret bound in 500 independently constructed vectors | No violation of \(2\epsilon\) |
| DR expectation with deliberately wrong outcome models and correct propensities | Exact expected value 0.8; numerical discrepancy zero |
| Grouped bootstrap | Repeating identical seeds within each problem does not increase independent problem count |
| Outcome-dependent replay stopping | A Bernoulli(0.5) stopped sample mean has expectation 0.625 under the stipulated rule |
| Cross-fit dependence | An exactly enumerated leave-one-out example has nonzero cross-problem covariance and zero coverage for its fixed-contribution percentile interval |
| Same-sample “oracle” | Two equal 0.5-success actions have expected maximum single-replay outcome 0.75; fresh evaluation of the selected action remains 0.5 |

An exactly enumerated sensor example has two observed strata. In one, continue already succeeds with probability 0.85. In the other, two equally likely latent states have action values \((.8,.2,.6)\) and \((.2,.8,.6)\), and a synthetic sensor distinguishes them with probability 0.9. Each stratum has weight 0.5 and acquisition costs 0.03 utility units. The cheap policy has value **0.725**, always acquiring the sensor has net value **0.765**, and acquiring only in the ambiguous stratum has net value **0.780**. These values follow from the constructed probabilities; they are not estimated real-world gains or a monetary conversion for Jev.

The script also checks a counterexample: a post-query action increases value from 0.50 to 0.54, but a query cost of 0.10 lowers its net value to 0.44. This is why Proposition 4 cannot justify a free-query assumption. Increasing the synthetic sensor cost to 0.20 makes no-query optimal in both strata.

The cross-fit counterexample uses two independent Bernoulli outcomes \(Y_1,Y_2\) with mean 0.5. A fold-trained policy chooses the positive-reward action for problem 1 only when \(Y_2=1\), and conversely for problem 2; the alternative yields zero reward. Each held-out contribution equals \(Y_1Y_2\). Thus the mean procedure value is 0.25, but every realized pair is either \((0,0)\) or \((1,1)\). Resampling those fixed contributions produces a point interval at zero or one, neither covering 0.25. This finite example refutes an unconditional bootstrap guarantee; it does not assert that all large-sample cross-fit inference fails.

Reproduce with `python scripts/validate_theory.py` and `python -m pytest tests/test_theory.py -q` from an environment with the project's test dependencies. No API key or model weights are used.
