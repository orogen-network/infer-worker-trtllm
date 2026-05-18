# infer-worker-trtllm

TensorRT-LLM backed operator daemon for the dc-premium tier (H100/H200 hosts running
quantized FP8 70B+ models with high tokens/s).

## Architecture

Mirrors `infer-worker-vllm` (FastAPI + signed receipts per RFC-0001 + heartbeat per
RFC-0003) but routes inference through an `Engine` Protocol with two implementations:

- **`MockTrtllmEngine`** — deterministic pseudo-tokens (prefix `"tr"`); used by tests.
- **`RealTrtllmEngine`** — wraps `tensorrt_llm.runtime.ModelRunner`. Raises
  `RuntimeError("TRT-LLM not available")` if the runtime isn't on PYTHONPATH (we use
  `RuntimeError`, not `ImportError`, because TRT-LLM also has CUDA/plugin-DLL deps
  that pip alone can't satisfy).

## TRT-LLM prod integration sketch

```python
from tensorrt_llm.runtime import ModelRunner
runner = ModelRunner.from_dir(
    engine_dir="/srv/engines/llama-3-70b-h100",
    rank=0,
    tp_size=4,
)
```

Operator must pre-compile the engine for the target hardware (`trtllm-build`).

## Endpoints

Same as `infer-worker-vllm`: `/v1/chat/completions`, `/healthz`, `/internal/last_heartbeat`.

## Tier semantics

This worker is advertised at `price_per_million_tokens = 4_000_000` (CUC units) and
declares `quantization = FP8` in its heartbeat capability so the gateway router can
classify it as dc-premium tier per RFC-0008 §5.
