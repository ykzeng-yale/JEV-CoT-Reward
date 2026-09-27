"""Prevent deployment data from changing the learned representation."""
import numpy as np
import pytest
from jev_control.representation import fit_representation


def test_frozen_transform_preserves_training_statistics_and_batch_independence():
    fitted, train = fit_representation(['alpha beta', 'beta gamma'], [[1, np.nan], [3, 4]])
    vocabulary = dict(fitted.vectorizer.vocabulary_)
    mean = fitted.scaler.mean_.copy()
    stats = fitted.imputer.statistics_.copy()
    alone = fitted.transform(['alpha unseenword'], [[2, np.nan]])
    together = fitted.transform(['alpha unseenword', 'unseenword newword'], [[2, np.nan], [1e8, -1e8]])
    np.testing.assert_allclose(alone.toarray(), together[:1].toarray())
    assert fitted.vectorizer.vocabulary_ == vocabulary
    assert 'unseenword' not in vocabulary
    np.testing.assert_array_equal(fitted.scaler.mean_, mean)
    np.testing.assert_array_equal(fitted.imputer.statistics_, stats)
    np.testing.assert_allclose(fitted.transform(['alpha beta', 'beta gamma'], [[1, np.nan], [3, 4]]).toarray(), train.toarray())


def test_empty_training_vocabulary_and_all_missing_column_keep_frozen_shape():
    fitted, train = fit_representation(['!', '?'], [[np.nan, 1], [np.nan, 3]])
    assert fitted.vocabulary_size == 0
    assert fitted.vectorizer is None
    future = fitted.transform(['novel text'], [[4, 5]])
    assert future.shape[1] == train.shape[1]
    assert np.isfinite(future.toarray()).all()


def test_schema_drift_and_infinities_are_rejected():
    fitted, _ = fit_representation(['alpha', 'beta'], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match='schema width'):
        fitted.transform(['alpha'], [[1]])
    with pytest.raises(ValueError, match='schema width'):
        fitted.transform(['alpha', 'beta'], [[1, 2]])
    with pytest.raises(ValueError, match='Infinite'):
        fitted.transform(['alpha'], [[np.inf, 2]])
    with pytest.raises(ValueError, match='Nonempty'):
        fit_representation([], [])
