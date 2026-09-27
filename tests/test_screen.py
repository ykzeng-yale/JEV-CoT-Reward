"""All-arm instrumentation tests with a deterministic token-ID fake backend."""
import importlib.util
from pathlib import Path

import pytest

from jev_control.mlx_backend import Generation


spec = importlib.util.spec_from_file_location("run_screen", Path(__file__).parents[1] / "scripts/run_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class FakeBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def encode_text(self, text):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def generate(self, prefix, max_tokens, seed, timeout_s):
        self.calls.append((list(prefix), max_tokens, seed))
        text, finish_reason = self.responses.pop(0)
        ids = self.encode_text(text)
        assert len(ids) <= max_tokens
        return Generation(ids, text, len(prefix), len(ids), -1.0, 1.0,
                          0.1, finish_reason, 0.0, "fake", seed)

    choose_candidate = staticmethod(lambda candidates: 0)


TASK = {"family": "weighted_path", "data": {"n": 2, "edges": [[0, 1, 1]]}}


def test_continue_uses_exact_prefix_and_charges_finalization():
    backend = FakeBackend([("work", "length"), ("A->B", "stop")])
    prompt = backend.encode_text("PROMPT")
    retained = backend.encode_text("observed\n\n")
    result = screen.rollout(backend, TASK, prompt, retained, "continue", 11, 40, 12)
    assert backend.calls[0][0] == prompt + retained
    assert backend.calls[0][1] == 28
    assert result["generated_tokens"] == 8
    assert result["outcome"]["success"]
    assert result["prompt_tokens_processed"] == sum(len(x[0]) for x in backend.calls)


def test_branch_charges_both_candidates_and_continues_only_winner():
    backend = FakeBackend([("first", "length"), ("other", "length"), ("FINAL: A->B", "stop")])
    prompt = backend.encode_text("PROMPT")
    retained = backend.encode_text("observed\n\n")
    result = screen.rollout(backend, TASK, prompt, retained, "branch", 11, 80, 12)
    assert backend.calls[0][0] == backend.calls[1][0] == prompt + retained
    assert backend.calls[2][0] == prompt + retained + backend.encode_text("first")
    assert result["generated_tokens"] == len("firstotherFINAL: A->B")
    assert result["outcome"]["success"]


def test_finished_candidate_is_not_continued_past_eos():
    backend = FakeBackend([("FINAL: A->B", "stop"), ("other", "length")])
    result = screen.rollout(backend, TASK, [], [], "branch", 11, 80, 12)
    assert len(backend.calls) == 2
    assert result["outcome"]["success"]


def test_boundary_returns_original_index_without_retokenizing():
    backend = FakeBackend([])
    tokens = backend.encode_text("first paragraph\n\nsecond unfinished")
    assert screen.boundary(backend, tokens, minimum=1) == len("first paragraph\n\n")
    assert screen.boundary(backend, backend.encode_text("no boundary"), minimum=1) is None


@pytest.mark.parametrize("partial,tail", [("FINAL: A->", "B"), ("FINAL:", " A->B")])
def test_final_reserve_completes_partial_marker_without_rewriting(partial, tail):
    backend = FakeBackend([(partial, "length"), (tail, "stop")])
    prefix = backend.encode_text("prompt")
    retained = backend.encode_text("observed\n\n")
    result = screen.rollout(backend, TASK, prefix, retained, "continue", 11, 40, 12)
    assert backend.calls[1][0] == prefix + retained + backend.encode_text(partial)
    assert backend.calls[1][1] == 40 - len(partial)
    assert not result["overhead"]
    assert result["outcome"]["success"]


@pytest.mark.parametrize("remaining,reserve,action", [(10, -1, "continue"), (10, 11, "continue"), (10, 10, "branch")])
def test_invalid_envelopes_rejected_before_generation(remaining, reserve, action):
    backend = FakeBackend([])
    with pytest.raises(ValueError):
        screen.rollout(backend, TASK, [], [], action, 11, remaining, reserve)
    assert not backend.calls


def test_retained_statistics_exclude_discarded_future():
    result = Generation([1, 2, 3], "", 0, 3, -34, 35, 0, "length", 0, "", 0,
                        token_logprobs=[-1, -1, -100], token_entropies=[2, 2, 101])
    stats = screen.retained_statistics(result, 2)
    assert stats == {"retained_stat_token_count": 2,
                     "retained_mean_logprob": -1, "retained_mean_entropy": 2}


def test_historical_generation_statistics_remain_missing():
    result = Generation([1, 2], "", 0, 2, -1, 1, 0, "length", 0, "", 0)
    stats = screen.retained_statistics(result, 2)
    assert stats["retained_mean_logprob"] is None
    assert stats["retained_mean_entropy"] is None


@pytest.mark.parametrize("finish,text,end,reason", [
    ("length", "FINAL: A->", 3, "answer_phase_reached_before_checkpoint"),
    ("stop", "FINAL: A->B", 3, "completed_before_checkpoint"),
    ("length", "work", None, "no_paragraph_boundary"),
    ("length", "work", 3, None),
])
def test_eligibility_never_confuses_answer_marker_with_completion(finish, text, end, reason):
    result = Generation([1, 2], text, 0, 2, -1, 1, 0, finish, 0, "", 0)
    assert screen.ineligible_reason(result, end) == reason
