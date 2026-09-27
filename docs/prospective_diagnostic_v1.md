# Prospective diagnostic after the negative mechanism screen

Design frozen 2026-09-27, before generation of the new seed range. This is a bounded diagnostic, **not** a passed pilot gate or a confirmatory efficacy trial. The larger campaign remains closed.

## Evidence and reason to run

The completed mechanism screen had no improvement from Jev, local semantic features, forests or sparse gates over continuation. Within-family Ridge success was 18.75% cheap versus 16.67% Jev on arithmetic, and 75% for both on graph tasks. Continue achieved 20.83% and 77.08%, respectively. Only four of 24 checkpoints had an alternative with an observed mean above continue, each by 0.25. Four repeats per action make this an unreliable way to select apparently responsive checkpoints. We therefore do not pick these four for favorable-case replication.

A small new-problem diagnostic answers an outstanding question: what happens when the fixed controllers actually select actions before any continuation outcome is available? It addresses prospective operation, not broad generalization, sequential control or a small effect with adequate power. Maximum new continuations are 96, below the 288-run screen. No intervention, prompt, difficulty, learner hyperparameter or feature schema is tuned in response to this test.

## Frozen comparison

Use [the configuration](../configs/prospective_diagnostic_v1.json). Fit the original per-action Ridge learner with alpha 10 on all 24 screen problems, retaining each Bernoulli outcome with the existing equal-problem weights. Fit TF-IDF, imputation and scaling on training problems only. Do not choose between Ridge, forest and sparse gates based on their results. Freeze and hash all fitted artifacts, training identities, source and package versions before generating any new test task. A generated-state fixture must verify the frozen predictor against its training implementation first.

Compare always-continue, cheap features, cheap plus Jev, and cheap plus the independently prompted local judge. Continue here includes the common online pause and finalization protocol; it is not a new uninterrupted baseline. There is one new checkpoint trajectory per problem, paired across policies. Use task seed 591027, indices 0–23, and the same generator, sampling, actions and token allowance as mechanism v1. All sampled problems remain in the denominator, including early completion or absent checkpoints. A timeout or malformed answer is not silently replaced.

Acquire features from the allowed observable state only. Local judging may be a separate bounded pass before continuation to avoid concurrent model loading. Both learned semantic policies use their own feature source; predictions never use outcome labels or the competing judge's features. Record all four action decisions durably before executing the first continuation at that problem. There is no repeated intervention.

If policies choose the same action, reuse that problem's prespecified action/seed continuation. This is an explicit paired common-result design: it reduces collection work without changing the marginal frozen-policy target. Different policies' results are dependent, and all inference remains problem-level. Charge the complete shared prefix, selected action and feature-acquisition cost to each hypothetical deployment episode, even when collection reuses work. Report actual collection cost separately. Feature caching is not free hypothetical deployment.

The primary contrast is Jev minus cheap verified final success. Report every policy, action rate, family, invalid answer, acquisition failure and cost. Use paired family-stratified problem bootstrap descriptively; n=24 is too small to establish equivalence, small benefits or arbitrary absence of value. Do not adapt enrollment or stop on favorable interim outcomes. A favorable result triggers review of uncertainty, not automatic expansion.

## Launch and completion requirements

Implementation is still outstanding. Before launch: verify model/runtime identity; train and freeze predictors; verify no overlap with prior task seeds; test prospective decision ordering, common-result accounting and early-finish inclusion; save the complete schedule; use the central Jev ledger with cumulative $1 cap; use bounded single-model execution. After completion: independently audit every selected action, call, final answer and resource counter before analysis. Preserve original labels and publish a reproducible report, including a negative result if observed.
