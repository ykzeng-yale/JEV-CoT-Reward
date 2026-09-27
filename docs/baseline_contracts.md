# Additional controller baselines and a binary-outcome distinction

Added during the exploratory mechanism collection, before its final analysis. These are independently written adaptations, not reproductions of published benchmark results. They do not alter the running generator, actions, rubric or outcomes. No main-test policy is selected here.

## Implemented comparisons

[Calibration Is Not Control](https://arxiv.org/html/2606.21399v1#S4.SS1) motivates per-action success estimation and a tree-dispersion penalty. Its Appendix B.5 specifies 200 trees, depth six and minimum leaf size two. Our forest uses those settings, our existing prefix features and actions, and a fixed penalty coefficient of 0.5. Both unpenalized and penalized selection are reported; neither is selected using the test results. This is an adaptation because tasks, utilities, features, action meanings and parameter selection differ. Tree dispersion is a heuristic, **not a calibrated confidence bound**, particularly when repeated outcomes share identical inputs.

[DIAL](https://arxiv.org/html/2605.06908v1#S4.SS3) motivates a sparse logistic intervention gate. We use two separately reported binary gates: continue versus repair, and continue versus branch. We fix C=1 and a strict probability threshold of 0.5, and adapt the training labels to the binary terminal-success objective as derived below. We do not reproduce DIAL's task-specific feature proposal, exploration trigger, tuning or online adaptation. Its main text describes threshold selection while Appendix C.1 fixes 0.5; our choice is explicit rather than treating these as identical protocols.

All methods use the same problem-grouped outer folds and training-only TF-IDF, numeric imputation and scaling as the Ridge comparison. Every continuation remains a Bernoulli observation for the forests. Sparse gates pair continue/alternative records by the prespecified repeat index; they never pair successful alternatives selectively with failed continuations. Distinct seeds do not imply common random numbers. Weights give each original problem equal total pair mass **before** discarding zero-loss ties. No class balancing changes the outcome target.

Each learner is crossed with cheap features, cheap-plus-Jev, and cheap-plus-local when those features exist. Compare within learner before attributing differences to Jev. The global and family-specific training-selected constants remain controls. We report all variants and family means; selecting the best observed result would require fresh validation. Cross-fit bootstrap ranges remain descriptive and have no generic nominal coverage because fitted folds overlap.

Additional fixed heuristics branch above the training-fold median entropy, or repair below a semantic local-validity score of 0.7; missing scores default to continue. Semantic heuristics use the collected seven-question batch and its measured cost. A dedicated one-question service could cost less, but that configuration is unmeasured. A rate-matched replay diagnostic averages permutations of each policy's selected actions within the same held-out fold and task family. It measures targeting relative to that restricted random assignment, not fresh randomized deployment. Actions are never permuted across folds, which could introduce outcome leakage.

The runner is `scripts/analyze_baselines.py`. It first invokes the strict screen analysis and refuses a missing/incomplete learned-comparison gate. It makes no model or API calls, saves per-problem decisions and source hashes, and refuses to overwrite an existing analysis. Run only after the independent mechanism audit and local-judge acquisition:

```sh
.venv/bin/python scripts/analyze_baselines.py runs/mechanism-v1-20260927 \
  --integrity-report results/mechanism_v1_integrity.json \
  --output results/mechanism_v1_additional_baselines.json
```

## Why ties matter for an intervention gate

This is an elementary specialization of binary policy learning, not a claimed new general theorem. Let Y_c,Y_a be paired binary terminal outcomes and W the decision features. No independence between the two outcomes is required. Define

\[
 p_+(w)=P(Y_a=1,Y_c=0\mid W=w),\quad
 p_-(w)=P(Y_a=0,Y_c=1\mid W=w).
\]

The expected intervention advantage is exactly

\[
 \Delta(w)=E[Y_a-Y_c\mid w]=p_+(w)-p_-(w).
\]

This follows by enumerating the four binary outcome pairs: the two ties contribute zero, the win contributes +1, and the loss contributes −1. Therefore predicting an outright win versus everything else and thresholding at 0.5 is not generally equivalent to choosing positive expected advantage.

For an illustrative fixed checkpoint with independent success probabilities 0.8 and 0.7, the alternative has advantage 0.1, but outright-win probability is only 0.24. This example is stipulated mathematics, not a measured Jev or model result.

Let r(w)=p_+(w)+p_-(w) be discordance probability. Where r(w)>0, set η(w)=p_+(w)/r(w). Then

\[
 \Delta(w)=r(w)(2\eta(w)-1).
\]

Thus η>0.5 has the correct sign for binary expected gain. At r=0, both actions have the same expected outcome; set η=0.5 by convention. Our sparse gate estimates this conditional probability, **not** terminal success probability or advantage magnitude. It cannot rank repair against branch by comparing their conditional-win probabilities; that is why the two gates are reported separately.

For any binary gate g(W), its decision loss on a paired observation is

\[
 \ell_g=\mathbf1\{Y_a>Y_c\}(1-g(W))+
          \mathbf1\{Y_a<Y_c\}g(W).
\]

Writing V(g)=E[Y_c+g(W)(Y_a-Y_c)] gives

\[
 E\ell_g=P(Y_a>Y_c)-[V(g)-E Y_c].
\]

Consequently, minimizing this weighted classification loss is equivalent to maximizing binary outcome value over the same gates. Dropping ties preserves the loss because they contribute zero; renormalizing each problem by its *discordant* count would change the intended weighting. The implementation keeps weights based on all pairs. Classification labels remain noisy samples, not declarations of a checkpoint's true optimal action.

## Surrogate loss and the limits of that implication

For a fitted probability q(w) in (0,1), let g_q=1{q>0.5} and g*=1{η>0.5}. Their value gap is

\[
 V(g^*)-V(g_q)=E[r(W)|2\eta(W)-1|\mathbf1\{g_q\ne g^*\}].
\]

Define population excess weighted logistic loss relative to the unrestricted true conditional probability as

\[
 \mathcal E_{\log}=E[r(W)\,\mathrm{KL}(\mathrm{Bern}(\eta(W))\Vert\mathrm{Bern}(q(W)))].
\]

Then

\[
 V(g^*)-V(g_q)\le\sqrt{2E[r(W)]\mathcal E_{\log}}.
\]

**Proof.** On a wrong-sign decision, |2η−1|≤2|η−q|. For fixed q, the function KL(Bern(p)∥Bern(q))−2(p−q)² has second derivative 1/[p(1−p)]−4≥0 and its value and derivative vanish at p=q; hence KL≥2(p−q)², with endpoint limits. The pointwise loss is therefore at most r√(2 KL). Cauchy–Schwarz yields the displayed bound. Zero-discordance states contribute zero. This proves a population implication, not a guarantee for a finite regularized fit.

Representation error, limited function class, finite replay noise, overlapping cross-fit training, distribution shift and acquisition costs all remain. C=1 is a declared baseline setting, not a verified optimal choice. A sign gate also ignores gain magnitude, making it insufficient for allocating a shared compute budget among several checkpoints. Fresh frozen-policy outcomes remain necessary.
