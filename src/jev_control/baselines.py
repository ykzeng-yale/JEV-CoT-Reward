"""Small paper-inspired control baselines; no inference or hosted API access.

Callers must fit representations on training problems only and keep all repeats
of a problem in one fold. Tree dispersion below is a heuristic penalty, NOT a
confidence interval. See docs/baseline_contracts.md for adaptations and limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def positive_probability(model, x):
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    return model.predict_proba(x)[:, matches[0]] if len(matches) else np.zeros(x.shape[0])


@dataclass
class ActionForest:
    models: list
    beta: float

    def predict(self, x):
        means, deviations = [], []
        for model in self.models:
            mean = positive_probability(model, x)
            if len(model.classes_) == 1:
                deviation = np.zeros(x.shape[0])
            else:
                # Tree labels encode the forest's sorted classes; binary 1 is
                # the success column when the forest has both classes.
                tree_values = np.asarray([positive_probability(tree, x) for tree in model.estimators_])
                deviation = tree_values.std(axis=0, ddof=0)
            means.append(mean)
            deviations.append(deviation)
        means = np.stack(means, axis=1)
        deviations = np.stack(deviations, axis=1)
        return {"success_probability": means, "tree_dispersion": deviations,
                "selection_score": means - self.beta * deviations}


def fit_action_forest(x, action, outcome, groups, *, beta=0.5, seed=20260927):
    """Three per-action forests using every Bernoulli continuation."""
    action, outcome, groups = np.asarray(action), np.asarray(outcome), np.asarray(groups)
    if not (len(action) == len(outcome) == len(groups) == x.shape[0]) or not len(action):
        raise ValueError("Aligned nonempty training rows required")
    if not math.isfinite(beta) or beta < 0 or not np.isin(outcome, [0, 1]).all():
        raise ValueError("Finite nonnegative penalty and binary outcomes required")
    if set(action) != {0, 1, 2}:
        raise ValueError("All three action codes are required")
    models = []
    for a in range(3):
        selected = np.flatnonzero(action == a)
        _, inverse, counts = np.unique(groups[selected], return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
        model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                       random_state=seed + a, n_jobs=1)
        model.fit(x[selected], outcome[selected], sample_weight=weights)
        models.append(model)
    return ActionForest(models, beta)


@dataclass
class SparsePairGate:
    model: object | None
    constant: float | None
    discordant_pairs: int
    total_pairs: int

    def predict(self, x):
        return np.full(x.shape[0], self.constant) if self.model is None else positive_probability(self.model, x)


def fit_sparse_pair_gate(x, continue_outcome, alternative_outcome, groups, *, c=1.0, seed=20260927):
    """Fit P(alternative wins | discordance,x); ties carry zero utility loss.

    Pairing is by a prespecified repeat index, not by choosing favorable
    continuations. This adapts a binary gate to our binary terminal endpoint;
    it is not an exact DIAL reproduction or an action-value probability.
    """
    base, alt, groups = np.asarray(continue_outcome), np.asarray(alternative_outcome), np.asarray(groups)
    if not (len(base) == len(alt) == len(groups) == x.shape[0]) or not len(base):
        raise ValueError("Aligned nonempty training pairs required")
    if not np.isin(base, [0, 1]).all() or not np.isin(alt, [0, 1]).all() or not math.isfinite(c) or c <= 0:
        raise ValueError("Binary outcomes and finite positive C required")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]  # Before removing ties; retain their frequency.
    discordant = base != alt
    labels = (alt[discordant] > base[discordant]).astype(int)
    if not len(labels):
        return SparsePairGate(None, 0.5, 0, len(base))  # Strict >.5 gate continues.
    if len(np.unique(labels)) == 1:
        return SparsePairGate(None, float(labels[0]), len(labels), len(base))
    regularizer = ({"l1_ratio": 1.0} if inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"
                   else {"penalty": "l1"})
    model = LogisticRegression(solver="liblinear", C=c, max_iter=2000, random_state=seed, **regularizer)
    model.fit(x[discordant], labels, sample_weight=weights[discordant])
    return SparsePairGate(model, None, len(labels), len(base))
