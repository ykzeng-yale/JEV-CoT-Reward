"""Frozen per-action Ridge predictor for outcome-grounded prospective control."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
from sklearn.linear_model import Ridge
from .representation import FrozenRepresentation, fit_representation

ACTIONS = ('continue', 'repair', 'branch')


@dataclass
class FrozenRidgeController:
    representation: FrozenRepresentation
    models: list
    training_problem_ids: tuple[str, ...]
    alpha: float

    def predict(self, documents, numeric):
        matrix = self.representation.transform(documents, numeric)
        return np.column_stack([np.clip(model.predict(matrix), 0, 1) for model in self.models])

    def choose(self, documents, numeric):
        predictions = self.predict(documents, numeric)
        return [ACTIONS[i] for i in predictions.argmax(axis=1)]


def fit_controller(documents, numeric, rows, problem_ids, *, alpha=10.):
    """Fit Bernoulli rows with the same normalized equal-problem weights as screening.

    Caller supplies training-only observable features; rows contain outcomes only
    for fitting. Deployment prediction accepts no outcome table or task answer.
    """
    ids = tuple(problem_ids)
    if len(set(ids)) != len(ids) or len(ids) != len(documents) or not ids:
        raise ValueError('Unique aligned training problem IDs required')
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Finite positive alpha required')
    locations = {pid: i for i, pid in enumerate(ids)}
    for row in rows:
        if row['problem_id'] not in locations or row['action'] not in ACTIONS:
            raise ValueError('Unknown training problem or action')
        if type(row['outcome']['success']) not in (bool, int) or row['outcome']['success'] not in (0, 1):
            raise ValueError('Binary training outcomes required')
    representation, matrix = fit_representation(documents, numeric)
    models = []
    for action in ACTIONS:
        selected = [row for row in rows if row['action'] == action]
        counts = Counter(row['problem_id'] for row in selected)
        if set(counts) != set(ids):
            raise ValueError('Every training problem requires every action')
        weights = np.asarray([1 / counts[row['problem_id']] for row in selected])
        weights *= len(weights) / weights.sum()
        model = Ridge(alpha=alpha, solver='lsqr')
        model.fit(matrix[[locations[row['problem_id']] for row in selected]],
                  [row['outcome']['success'] for row in selected], sample_weight=weights)
        models.append(model)
    return FrozenRidgeController(representation, models, ids, alpha)
