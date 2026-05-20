"""Worker configuration (TRT-LLM variant)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerConfig:
    operator_id: str
    operator_private_key_hex: str
    gateway_id: str
    attestation_report_hash: str
    model_id: str = "mock-trtllm-70b"
    model_weight_hash: str = "0x" + "ab" * 32
    kernel_pack_hash: str = "0x" + "cd" * 32
    heartbeat_interval_s: float = 12.0
    base_url: str = ""
    capabilities: list[str] = field(default_factory=lambda: ["mock-trtllm-70b"])
    gateway_auth_token: str = ""
    deterministic_mode: bool = True
    # TRT-LLM is the dc-premium tier; advertised price reflects that.
    price_per_million_tokens: int = 4_000_000
    # FP8 default for H100/H200 dc tier.
    quantization: str = "FP8"
    # On-disk path to the engine dir (TRT-LLM compiled engine) or weights
    # directory. If unset, the weight-hash verification step at startup is
    # skipped (Mock engines have no weights to verify). See
    # `weights.verify_weights`.
    model_path: str | None = None
