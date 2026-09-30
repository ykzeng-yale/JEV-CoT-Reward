"""Hardware-stress tests need no CUDA/model and exercise envelope/provenance guards."""
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "recoma_context_stress_v3", Path(__file__).parents[1] / "scripts/recoma_context_stress_v3.py"
)
stress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stress)


def test_exact_context_preserves_template_and_charges_padding():
    base = list(range(100))
    ids = stress.exact_context_ids(base, [901, 902, 903], 32768)
    assert len(ids) == 32768
    assert ids[:68] == base[:68]
    assert ids[-32:] == base[-32:]
    assert set(ids[68:-32]) == {901, 902, 903}
    assert stress.exact_context_ids(base, [901], 100) == base


@pytest.mark.parametrize("target", [0, -1, True, 1.5])
def test_invalid_context_budget(target):
    with pytest.raises(ValueError):
        stress.exact_context_ids([1, 2], [3], target)


def test_no_silent_truncation_or_invalid_ids():
    with pytest.raises(ValueError, match="truncate"):
        stress.exact_context_ids([1, 2, 3], [4], 2)
    for base, filler in [([], [1]), ([1], []), ([True], [2]), ([1], [-2])]:
        with pytest.raises(ValueError):
            stress.exact_context_ids(base, filler, 10)


def test_existing_record_protected_and_hard_cap_before_model_access(tmp_path):
    p = tmp_path / "stress.json"
    p.write_text("original")
    with pytest.raises(FileExistsError):
        stress.run_context_stress(None, p)
    assert p.read_text() == "original"
    with pytest.raises(ValueError, match="envelope"):
        stress.run_context_stress(None, tmp_path / "large.json", target_tokens=65537)
    assert not (tmp_path / "large.json").exists()


def test_failure_record_is_separate_and_durable_without_model_loading(tmp_path, monkeypatch):
    # CUDA-unavailable failure must retain scope and must not create a model.
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)))
    generator = SimpleNamespace(model_id=stress.MODEL, model_revision=stress.REVISION)
    path = tmp_path / "stress.json"
    with pytest.raises(RuntimeError, match="allocated"):
        stress.run_context_stress(generator, path)
    report = json.loads(path.read_text())
    assert report["status"] == "infrastructure_failure"
    assert report["scientific_episode"] is False
    assert report["model_reload"] is False
    assert report["generated_tokens"] == 0
    assert report["scope"] == "synthetic_hardware_only_not_scientific_outcome"


def test_full_stress_envelope_and_controller_config_unchanged(tmp_path, monkeypatch):
    import sys

    class Tensor:
        def __init__(self, ids):
            self.ids = ids

        def __getitem__(self, key):
            _row, selection = key
            return SimpleNamespace(tolist=lambda: self.ids[selection])

    class Tokenizer:
        pad_token_id = 77

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return list(range(80))

        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return [901, 902]

        def decode(self, ids, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return "hardware tokens " + str(len(ids))

    class Model:
        dtype = "bf16"
        config = SimpleNamespace(_attn_implementation="sdpa", max_position_embeddings=65536)
        generation_config = SimpleNamespace(eos_token_id=123, forced_eos_token_id=124,
                                            stop_strings=["original-stop"], do_sample=True,
                                            pad_token_id=125)

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == kwargs["min_new_tokens"] == 400
            assert kwargs["do_sample"] is False and kwargs["use_cache"] is True
            config = kwargs["generation_config"]
            assert config.eos_token_id is None and config.forced_eos_token_id is None
            assert config.stop_strings is None and config.pad_token_id == 77
            assert len(kwargs["input_ids"].ids) == 32768
            return Tensor(kwargs["input_ids"].ids + [7] * 400)

    fake_torch = SimpleNamespace(
        bfloat16="bf16", long="long",
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             get_device_name=lambda i: "Fake allocated GPU",
                             synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                             max_memory_allocated=lambda: 123456,
                             max_memory_reserved=lambda: 234567),
        tensor=lambda ids, **kwargs: Tensor(ids[0]), ones_like=lambda tensor: tensor,
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = Model()
    generator = SimpleNamespace(model=model, tokenizer=Tokenizer(), model_id=stress.MODEL,
                                model_revision=stress.REVISION)
    path = tmp_path / "stress.json"
    record = stress.run_context_stress(generator, path)
    stored = json.loads(path.read_text())
    assert stored == record
    assert record["status"] == "completed"
    assert record["input_tokens"] == record["processed_prompt_tokens"] == 32768
    assert record["output_tokens"] == record["generated_tokens"] == 400
    assert record["full_completion_envelope_reached"] is True
    assert record["model_calls"] == 1 and record["scientific_episode"] is False
    assert record["elapsed_seconds"] > 0
    assert record["output_text"] == "hardware tokens 400"
    assert record["peak_allocated_gpu_bytes"] == 123456
    assert model.generation_config.eos_token_id == 123
    assert model.generation_config.stop_strings == ["original-stop"]
