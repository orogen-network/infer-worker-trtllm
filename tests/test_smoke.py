"""infer-worker-trtllm tests."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from mining_types import LoadSnapshot, Receipt

from infer_worker_trtllm import (
    InferenceResult,
    MockTrtllmEngine,
    RealTrtllmEngine,
    WorkerConfig,
    build_app,
)
from infer_worker_trtllm.heartbeat import build_heartbeat
from infer_worker_trtllm.weights import verify_weights


def _nonce() -> str:
    return "0x" + secrets.token_hex(32)


@pytest.fixture
def config() -> WorkerConfig:
    return WorkerConfig(
        operator_id="op-trt",
        operator_private_key_hex="33" * 32,
        gateway_id="gw-test",
        attestation_report_hash="aa" * 32,
    )


def test_mock_engine_is_deterministic(config: WorkerConfig) -> None:
    e = MockTrtllmEngine(config.model_id)
    r1 = e.generate("hello trt", seed=0)
    r2 = e.generate("hello trt", seed=0)
    assert r1.text == r2.text
    assert r1.log_probs == r2.log_probs
    assert r1.tokens[0].startswith("tr")


def test_real_engine_raises_runtime_error_without_trtllm() -> None:
    with pytest.raises(RuntimeError, match="TRT-LLM not available"):
        RealTrtllmEngine("/nonexistent/engine", "mock-trtllm-70b")


def test_healthz_reports_engine_and_quant(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["operator_id"] == "op-trt"
        assert body["engine"] == "MockTrtllmEngine"
        assert body["quantization"] == "FP8"


def test_chat_completions_emits_signed_receipt(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 16,
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 200
        payload = r.json()
        rec = Receipt.model_validate(payload["receipt"])
        assert rec.operator_id == "op-trt"
        assert rec.operator_signature


def test_chat_rejects_unknown_model(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "nope",
                "messages": [{"role": "user", "content": "x"}],
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 400


def test_chat_rejects_missing_customer_nonce(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 422


def test_chat_rejects_replayed_customer_nonce(config: WorkerConfig) -> None:
    app = build_app(config)
    nonce = _nonce()
    with TestClient(app) as client:
        body = {
            "model": config.model_id,
            "messages": [{"role": "user", "content": "x"}],
            "customer_nonce": nonce,
        }
        r1 = client.post("/v1/chat/completions", json=body)
        assert r1.status_code == 200, r1.text
        r2 = client.post("/v1/chat/completions", json=body)
        assert r2.status_code == 409


def test_custom_engine_injection(config: WorkerConfig) -> None:
    @dataclass
    class FixedEngine:
        model_id: str = "mock-trtllm-70b"

        def generate(
            self, prompt: str, *, max_tokens: int = 32, seed: int = 0,
        ) -> InferenceResult:
            return InferenceResult(
                text="trt-canned",
                tokens=["trt-canned"],
                log_probs=[-0.7],
                prompt_tokens=2,
                completion_tokens=1,
            )

    app = build_app(config, engine=FixedEngine())
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": config.model_id,
                "messages": [{"role": "user", "content": "x"}],
                "customer_nonce": _nonce(),
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "trt-canned"


def test_heartbeat_declares_fp8(config: WorkerConfig) -> None:
    hb = build_heartbeat(config, LoadSnapshot())
    assert hb.capabilities[0].quantization.value == "FP8"
    assert hb.price_per_million_tokens == 4_000_000
    assert hb.signature


def test_verify_weights_detects_mismatch(tmp_path, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    w = tmp_path / "engine.bin"
    w.write_bytes(b"trt-engine")
    config.model_path = str(w)
    config.model_weight_hash = "0x" + ("33" * 32)
    with pytest.raises(RuntimeError, match="weight hash mismatch"):
        verify_weights(config)


def test_verify_weights_accepts_match(tmp_path, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    w = tmp_path / "engine.bin"
    w.write_bytes(b"trt-engine")
    config.model_path = str(w)
    config.model_weight_hash = "0x" + hashlib.sha256(b"trt-engine").hexdigest()
    verify_weights(config)


def test_verify_weights_refuses_placeholder_in_prod(monkeypatch, config: WorkerConfig) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.delenv("OROGEN_WORKER_SKIP_WEIGHT_CHECK", raising=False)
    with pytest.raises(RuntimeError, match="placeholder default"):
        verify_weights(config)


def test_last_heartbeat_endpoint(config: WorkerConfig) -> None:
    app = build_app(config)
    with TestClient(app) as client:
        r = client.get("/internal/last_heartbeat")
        assert r.status_code == 200
        assert r.json()["operator_id"] == "op-trt"
