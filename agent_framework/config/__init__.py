"""
ShieldPoint Configuration Loader
================================

Centralized config reader that pulls values from environment variables with
sensible defaults. Used by the agent framework and the test scripts.

Env vars are read at instantiation time (NOT at class-definition time) so
tests that monkeypatch env vars work correctly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LangfuseConfig:
    """Configuration for the Langfuse tracer.

    Construct via ``LangfuseConfig.from_env()`` to read from environment
    variables at call time. Direct construction with explicit kwargs is
    also supported (useful for tests).
    """
    host: str
    public_key: str
    secret_key: str
    enabled: bool
    flush_at: int
    flush_interval_ms: int
    retention_days: int

    @classmethod
    def from_env(cls) -> "LangfuseConfig":
        return cls(
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            enabled=os.environ.get("LANGFUSE_ENABLED", "true").lower() in (
                "1", "true", "yes", "on"
            ),
            flush_at=int(os.environ.get("LANGFUSE_FLUSH_AT", "15")),
            flush_interval_ms=int(os.environ.get("LANGFUSE_FLUSH_INTERVAL_MS", "1000")),
            retention_days=int(os.environ.get("LANGFUSE_RETENTION_DAYS", "90")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.public_key and self.secret_key and self.enabled)


@dataclass
class LMStudioConfig:
    """Configuration for the LM Studio inference endpoint."""
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LMStudioConfig":
        return cls(
            base_url=os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
            model=os.environ.get("QWEN_MODEL_ID", "qwen3.6-35b-a3b-q4_k_m"),
        )

    @property
    def host_url(self) -> str:
        """Base URL without /v1 suffix (for health checks)."""
        return self.base_url.rstrip("/").removesuffix("/v1")


@dataclass
class ShieldPointConfig:
    """Top-level configuration container."""
    langfuse: LangfuseConfig
    lm_studio: LMStudioConfig
    environment: str
    log_level: str


def load_config() -> ShieldPointConfig:
    """Load configuration from environment variables (read at call time)."""
    return ShieldPointConfig(
        langfuse=LangfuseConfig.from_env(),
        lm_studio=LMStudioConfig.from_env(),
        environment=os.environ.get("SHIELDPOINT_ENV", "development"),
        log_level=os.environ.get("SHIELDPOINT_LOG_LEVEL", "INFO"),
    )
