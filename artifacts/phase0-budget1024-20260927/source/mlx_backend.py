"""Bounded single-sequence MLX generation from exact token prefixes.

Validated against the installed mlx-lm 0.31.3 source. Loading requires a local
snapshot; this module never initiates a model download. Continuations do not
decode/re-encode their prefix or reapply a chat template. KV is recomputed from
the saved tokens so arms cannot contaminate one another through mutable caches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Sequence


_GENERATION_LOCK = threading.Lock()  # MLX RNG and tokenizer detokenizer are shared.


@dataclass(frozen=True)
class Generation:
    token_ids: list[int]
    text: str
    prompt_tokens: int
    generated_tokens: int
    mean_logprob: float | None
    mean_entropy: float | None
    elapsed_seconds: float
    finish_reason: str
    peak_memory_gb: float
    prefix_sha256: str
    seed: int
    token_logprobs: list[float] = field(default_factory=list)
    token_entropies: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLXBackend:
    """Frozen local model; cooperative timeout checked after each token.

    Timeout is not a hard OS deadline: prefill and a Metal kernel already in
    flight cannot be interrupted. The underlying MLX generator may evaluate one
    look-ahead token; elapsed time includes this compute but generated_tokens is
    the number of emitted tokens, including EOS. A subprocess deadline should
    wrap an entire run when a hard wall-clock limit is required.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_context_tokens: int = 16384,
        quantization_bits: int | None = None,
        quantization_group_size: int = 64,
    ) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("model_path must be an existing local model snapshot")
        if temperature < 0 or not 0 < top_p <= 1 or max_context_tokens <= 0:
            raise ValueError("Invalid sampling or context limit")
        if quantization_bits not in {None, 4, 8} or quantization_group_size <= 0:
            raise ValueError("Only optional 4/8-bit affine quantization is supported")
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self.mx = mx
        self._stream_generate = stream_generate
        self.model, self.tokenizer, self.model_config = load(
            str(path), lazy=quantization_bits is not None, return_config=True
        )
        if quantization_bits is not None:
            from mlx_lm.utils import quantize_model
            if "quantization" in self.model_config:
                raise ValueError("Refusing to requantize an already quantized snapshot")
            self.model, self.model_config = quantize_model(
                self.model, self.model_config, quantization_group_size,
                quantization_bits, mode="affine"
            )
            mx.eval(self.model.parameters())
            mx.clear_cache()
        self.quantization_config = self.model_config.get("quantization")
        self.model_path = str(path)
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_tokens = max_context_tokens
        self._sampler = make_sampler(temp=temperature, top_p=top_p)

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        return list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        ))

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def generate(
        self,
        prefix_ids: Sequence[int],
        max_tokens: int,
        seed: int,
        timeout_s: float = 120.0,
    ) -> Generation:
        prefix = list(prefix_ids)
        if not prefix or any(type(t) is not int or t < 0 for t in prefix):
            raise ValueError("prefix_ids must contain nonnegative integer token IDs")
        if type(max_tokens) is not int or max_tokens < 0 or timeout_s <= 0:
            raise ValueError("max_tokens must be nonnegative; timeout_s must be positive")
        if len(prefix) + max_tokens > self.max_context_tokens:
            raise ValueError("Requested prefix + output exceeds context envelope")
        prefix_hash = hashlib.sha256(
            json.dumps(prefix, separators=(",", ":")).encode()
        ).hexdigest()
        tokens: list[int] = []
        logprob_sum = 0.0
        entropy_sum = 0.0
        token_logprobs: list[float] = []
        token_entropies: list[float] = []
        finish_reason = "length"
        with _GENERATION_LOCK:
            start = time.perf_counter()
            self.mx.random.seed(seed)
            if max_tokens:
                stream = self._stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prefix,
                    max_tokens=max_tokens,
                    sampler=self._sampler,
                    # mlx-lm 0.31.3 otherwise normalizes BF16 logits in BF16;
                    # high logits can round selected-token logprobs to zero.
                    logits_processors=[lambda _ids, logits: logits.astype(self.mx.float32)],
                    prefill_step_size=512,
                )
                try:
                    for response in stream:
                        token = int(response.token)
                        tokens.append(token)
                        lp = response.logprobs.astype(self.mx.float32)
                        probs = self.mx.exp(lp)
                        entropy = -self.mx.sum(self.mx.where(probs > 0, probs * lp, 0))
                        selected_logprob = float(lp[token].item())
                        token_entropy = float(entropy.item())
                        token_logprobs.append(selected_logprob)
                        token_entropies.append(token_entropy)
                        logprob_sum += selected_logprob
                        entropy_sum += token_entropy
                        if response.finish_reason is not None:
                            finish_reason = response.finish_reason
                            break
                        if time.perf_counter() - start >= timeout_s:
                            finish_reason = "timeout"
                            break
                finally:
                    stream.close()
            elapsed = time.perf_counter() - start
            peak = self.mx.get_peak_memory() / 1e9
            text = self.decode(tokens)
        return Generation(
            token_ids=tokens,
            text=text,
            prompt_tokens=len(prefix),
            generated_tokens=len(tokens),
            mean_logprob=logprob_sum / len(tokens) if tokens else None,
            mean_entropy=entropy_sum / len(tokens) if tokens else None,
            elapsed_seconds=elapsed,
            finish_reason=finish_reason,
            peak_memory_gb=peak,
            prefix_sha256=prefix_hash,
            seed=seed,
            token_logprobs=token_logprobs,
            token_entropies=token_entropies,
        )

    @staticmethod
    def choose_candidate(candidates: Sequence[Generation]) -> int:
        """Fixed, gold-free branch selector: mean base-model token logprob.

        This is a likelihood heuristic, not a correctness verifier. Candidate
        generation costs must all be charged by the caller. Ties select the
        first candidate. Empty or nonfinite candidates receive negative infinity.
        """
        if not candidates:
            raise ValueError("At least one candidate is required")

        def score(index: int) -> float:
            value = candidates[index].mean_logprob
            return value if value is not None and math.isfinite(value) else -math.inf

        return max(range(len(candidates)), key=score)
