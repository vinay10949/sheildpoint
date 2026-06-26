"""
Configuration for the agent framework.

Reads from environment variables at call time (not at module-import time) so
tests that monkeypatch env vars work correctly. Mirrors the pattern already
used by ``agent_framework/config.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration for an :class:`Agent`.

    Construct via :meth:`from_env` in production, or directly in tests.
    """

    # ---- LM Studio --------------------------------------------------------
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"
    model: str = "qwen3.6-35b-a3b-q4_k_m"

    # ---- LLM call behaviour ----------------------------------------------
    #: Per-call timeout in seconds. AC: "if LLM call fails or times out
    #: (>10s), rule-based fallback executes and logs reason".
    llm_timeout_sec: float = 10.0

    #: Temperature for LLM calls. ReAct reasoning benefits from low
    #: temperature — keep at 0.1 by default.
    temperature: float = 0.1

    #: Maximum tokens for a single LLM reasoning step.
    max_tokens_per_step: int = 1024

    # ---- ReAct loop -------------------------------------------------------
    #: Hard cap on iterations before the loop is forcibly ended. Prevents
    #: runaway agent runs. SHLD-14 AC: max 10 iterations per claim.
    max_react_iterations: int = 10

    #: How many times to retry an LLM call that returns unparseable JSON
    #: before falling back.
    parse_retries: int = 1

    # ---- Confidence thresholds (SHLD-14) -----------------------------------
    #: Confidence threshold for HITL escalation. If the LLM's confidence
    #: is below this value, the claim is routed to human review.
    hitl_confidence_threshold: float = 0.85

    #: Confidence threshold for rule-based fallback. If the LLM's
    #: confidence is below this value, the FallbackEngine takes over
    #: entirely (no HITL queue — deterministic rules apply).
    fallback_confidence_threshold: float = 0.50

    #: Number of consecutive consistent LLM outputs required to confirm
    #: a decision. Used by the confidence scorer to measure stability.
    consistency_samples: int = 2

    # ---- Langfuse ---------------------------------------------------------
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = True

    # ---- Agent metadata ---------------------------------------------------
    environment: str = "development"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Read configuration from environment variables (at call time)."""
        return cls(
            lm_studio_base_url=os.environ.get(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            ),
            lm_studio_api_key=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
            model=os.environ.get("QWEN_MODEL_ID", "qwen3.6-35b-a3b-q4_k_m"),
            llm_timeout_sec=float(os.environ.get("LLM_TIMEOUT_SEC", "10")),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.1")),
            max_tokens_per_step=int(os.environ.get("LLM_MAX_TOKENS_PER_STEP", "1024")),
            max_react_iterations=int(os.environ.get("MAX_REACT_ITERATIONS", "10")),
            parse_retries=int(os.environ.get("LLM_PARSE_RETRIES", "1")),
            hitl_confidence_threshold=float(
                os.environ.get("HITL_CONFIDENCE_THRESHOLD", "0.85")
            ),
            fallback_confidence_threshold=float(
                os.environ.get("FALLBACK_CONFIDENCE_THRESHOLD", "0.50")
            ),
            consistency_samples=int(os.environ.get("CONSISTENCY_SAMPLES", "2")),
            langfuse_host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            langfuse_enabled=os.environ.get("LANGFUSE_ENABLED", "true").lower()
            in ("1", "true", "yes", "on"),
            environment=os.environ.get("SHIELDPOINT_ENV", "development"),
            log_level=os.environ.get("SHIELDPOINT_LOG_LEVEL", "INFO"),
        )

    @property
    def langfuse_configured(self) -> bool:
        """True iff we have enough Langfuse credentials to actually send traces."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )
