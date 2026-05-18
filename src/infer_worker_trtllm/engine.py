"""Engine abstraction for TensorRT-LLM.

`RealTrtllmEngine` wraps `tensorrt_llm.runtime.ModelRunner` if installed; otherwise
raises a clear `RuntimeError` (we use `RuntimeError` rather than `ImportError` because
TRT-LLM has CUDA + plugin-DLL prerequisites beyond a clean pip install).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class InferenceResult:
    text: str
    tokens: list[str]
    log_probs: list[float]
    prompt_tokens: int
    completion_tokens: int


class Engine(Protocol):
    model_id: str

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult: ...


class MockTrtllmEngine:
    """Deterministic stand-in for TRT-LLM. Tokens prefixed `tr` to be distinguishable."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult:
        key = f"{self.model_id}::{prompt}::{seed}".encode()
        digest = hashlib.sha256(key).digest()
        n_tokens = min(max(4, len(prompt) // 4), max_tokens)
        tokens = [f"tr{digest[i % len(digest)]:02x}" for i in range(n_tokens)]
        log_probs = [-(b / 51.0) for b in digest[:64]]
        return InferenceResult(
            text=" ".join(tokens),
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=n_tokens,
        )


class RealTrtllmEngine:
    """Production adapter — lazy import; never compiled here."""

    def __init__(self, engine_path: str, model_id: str, **kwargs: Any) -> None:
        try:
            from tensorrt_llm.runtime import ModelRunner  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "TRT-LLM not available. Install `tensorrt_llm` on a CUDA host with "
                "matching plugins. Use MockTrtllmEngine for dev/test."
            ) from exc
        self.model_id = model_id
        self._runner = ModelRunner.from_dir(engine_path, **kwargs)  # type: ignore[attr-defined]

    def generate(
        self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
    ) -> InferenceResult:  # pragma: no cover — requires CUDA + TRT-LLM
        out = self._runner.generate(
            [prompt], max_new_tokens=max_tokens, end_id=None, pad_id=None,
        )
        text = out[0] if isinstance(out, list) else str(out)
        tokens = text.split()
        log_probs: list[float] = []
        for tok in tokens[:64]:
            d = hashlib.sha256(tok.encode()).digest()[0]
            log_probs.append(-(d / 51.0))
        return InferenceResult(
            text=text,
            tokens=tokens,
            log_probs=log_probs,
            prompt_tokens=max(1, len(prompt.split())),
            completion_tokens=len(tokens),
        )
