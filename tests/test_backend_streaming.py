"""Exercise the real streaming loop without importing MLX or loading a model."""
from types import SimpleNamespace

import numpy as np
import pytest

import jev_control.mlx_backend as backend_module
from jev_control.mlx_backend import MLXBackend


class Stream:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.emitted = []
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        response = next(self.responses)
        self.emitted.append(response.token)
        return response

    def close(self):
        self.closed = True


def make_backend(finish_reasons=(None, None, "length")):
    stream = Stream([
        SimpleNamespace(token=i, finish_reason=finish,
                        logprobs=np.log(np.array([.4, .3, .2, .1])))
        for i, finish in enumerate(finish_reasons)
    ])
    backend = object.__new__(MLXBackend)
    backend.mx = SimpleNamespace(
        random=SimpleNamespace(seed=lambda _: None), float32=np.float32,
        exp=np.exp, sum=np.sum, where=np.where, get_peak_memory=lambda: 0,
    )
    backend.model = object()
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **_: "".join(map(str, ids)))
    backend.max_context_tokens = 100
    backend._sampler = object()
    requests = []

    def generate(*args, **kwargs):
        requests.append(kwargs)
        return stream

    backend._stream_generate = generate
    return backend, stream, requests


def test_online_stop_keeps_exact_prefix_and_never_reads_future_tokens():
    backend, stream, requests = make_backend()
    observed = []

    def stop(ids):
        observed.append(list(ids))
        return len(ids) == 2

    result = backend.generate([7, 8, 9], 3, 123, stop_when=stop)
    assert requests[0]["prompt"] == [7, 8, 9]
    assert requests[0]["max_tokens"] == 3
    assert observed == [[0], [0, 1]]
    assert stream.emitted == result.token_ids == [0, 1]
    assert result.finish_reason == "checkpoint"
    assert result.generated_tokens == len(result.token_logprobs) == len(result.token_entropies) == 2
    assert result.mean_logprob == pytest.approx(np.mean(np.log([.4, .3])))
    assert stream.closed


def test_callback_cannot_mutate_retained_ids_or_later_observations():
    backend, stream, _ = make_backend()
    observed = []

    def stop(ids):
        observed.append(list(ids))
        ids.append(99)
        return len(observed) == 2

    result = backend.generate([8], 3, 1, stop_when=stop)
    assert observed == [[0], [0, 1]]
    assert result.token_ids == [0, 1]
    assert stream.closed


def test_boundary_on_last_allowed_token_is_still_a_checkpoint():
    backend, stream, _ = make_backend()
    result = backend.generate([8], 3, 1, stop_when=lambda ids: len(ids) == 3)
    assert result.token_ids == [0, 1, 2]
    assert result.finish_reason == "checkpoint"
    assert stream.closed


def test_eos_is_charged_and_is_not_reclassified_as_checkpoint():
    backend, stream, _ = make_backend((None, "stop", None))
    observed = []
    result = backend.generate([8], 3, 1,
                              stop_when=lambda ids: observed.append(ids) or len(ids) == 2)
    assert result.finish_reason == "stop"
    assert result.token_ids == [0, 1]
    assert result.generated_tokens == 2
    assert observed == [[0]]
    assert stream.emitted == [0, 1]
    assert stream.closed


def test_timeout_retains_last_emitted_token_and_closes_stream(monkeypatch):
    backend, stream, _ = make_backend()
    clock = iter([10.0, 12.0, 12.5])
    monkeypatch.setattr(backend_module.time, "perf_counter", lambda: next(clock))
    result = backend.generate([8], 3, 1, timeout_s=1)
    assert result.finish_reason == "timeout"
    assert result.token_ids == [0]
    assert result.generated_tokens == 1
    assert result.elapsed_seconds == 2.5
    assert stream.closed


def test_callback_error_still_closes_stream():
    backend, stream, _ = make_backend()

    def fail(_):
        raise RuntimeError("invalid callback")

    with pytest.raises(RuntimeError, match="invalid callback"):
        backend.generate([8], 3, 1, stop_when=fail)
    assert stream.closed


def test_zero_budget_does_not_open_stream_or_invoke_callback():
    backend, stream, requests = make_backend()
    result = backend.generate([8], 0, 1, stop_when=lambda _: pytest.fail("No emitted token"))
    assert not requests
    assert result.generated_tokens == 0
    assert result.token_ids == []
    assert result.mean_logprob is None
    assert result.mean_entropy is None
