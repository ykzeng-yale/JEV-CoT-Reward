"""Backend contract checks that require neither MLX nor a loaded model."""
from dataclasses import replace

import pytest

from jev_control.mlx_backend import Generation, MLXBackend


def generation(score=None):
    return Generation([], "", 1, 0, score, None, 0.0, "length", 0.0, "", 1)


def test_selector_uses_finite_mean_likelihood_with_stable_ties():
    assert MLXBackend.choose_candidate([generation(-2), generation(-1)]) == 1
    assert MLXBackend.choose_candidate([generation(-1), generation(-1)]) == 0
    assert MLXBackend.choose_candidate([generation(), generation(float("nan")), generation(-5)]) == 2


def test_selector_rejects_empty_set():
    with pytest.raises(ValueError):
        MLXBackend.choose_candidate([])


def test_backend_never_downloads_a_nonlocal_model(tmp_path):
    with pytest.raises(ValueError, match="local model snapshot"):
        MLXBackend(str(tmp_path / "missing"))


def test_generation_serializes_raw_ids_and_counts():
    result = replace(generation(), token_ids=[3, 4, 5], generated_tokens=3)
    record = result.to_dict()
    assert record["token_ids"] == [3, 4, 5]
    assert record["generated_tokens"] == 3
