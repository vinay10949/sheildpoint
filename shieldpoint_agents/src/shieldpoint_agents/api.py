"""
Minimal FastAPI server exposing the agent framework.

Endpoints
---------

- ``GET /health`` — used by the Docker HEALTHCHECK and the smoke test.
  Returns ``{"status": "ok", "lm_studio": bool, "langfuse": bool}``.
- ``POST /run`` — runs a single claim through the agent. Accepts a claim
  dict in the body; returns the :class:`AgentRunResult` envelope.

This is intentionally a thin server — it lets the smoke test verify the
container is wired correctly to LM Studio and Langfuse without needing a
full backend. In production, the agent framework is meant to be embedded
in your application code, not deployed as a standalone service.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import (
    Agent,
    AgentConfig,
    FallbackEngine,
    LangfuseTracer,
    ToolRegistry,
)
from ._lmstudio import build_lm_studio_client, ping

logger = logging.getLogger("shieldpoint_agents.api")

app = FastAPI(
    title="ShieldPoint Agent Framework",
    version="0.1.0",
    description="Base agent framework for claims automation.",
)

# ---------------------------------------------------------------------------
# Build a default agent at startup. In a real deployment, you'd swap this
# for your own subclass with custom tools.
# ---------------------------------------------------------------------------
def _build_default_agent() -> Agent:
    registry = ToolRegistry()

    @registry.register(
        name="validate_policy",
        description="Look up a policy by ID and return its coverage limits.",
        schema={
            "type": "object",
            "properties": {"policy_id": {"type": "string"}},
            "required": ["policy_id"],
            "additionalProperties": False,
        },
    )
    def validate_policy(policy_id: str) -> dict[str, Any]:
        # Placeholder implementation — replace with real DB lookup.
        return {
            "policy_id": policy_id,
            "limit": 25_000,
            "deductible": 1_000,
            "perils_covered": ["wind", "hail", "fire"],
            "perils_excluded": ["flood", "earthquake"],
        }

    return Agent(
        name="claim-classifier",
        tools=registry,
        tracer=LangfuseTracer(agent_name="claim-classifier"),
        fallback=FallbackEngine(),
        config=AgentConfig.from_env(),
    )


_agent: Agent | None = None


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = _build_default_agent()
    return _agent


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    claim: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Health probe — verifies LM Studio is reachable."""
    cfg = AgentConfig.from_env()
    lm_ok = False
    try:
        client = build_lm_studio_client(cfg, timeout=2.0)
        lm_ok = ping(client)
    except Exception as exc:
        logger.debug("LM Studio ping failed: %s", exc)

    tracer = _get_agent().tracer
    return {
        "status": "ok",
        "lm_studio": lm_ok,
        "langfuse": tracer.enabled,
        "langfuse_disabled_reason": tracer.disabled_reason,
        "model": cfg.model,
    }


@app.post("/run")
def run_claim(req: RunRequest) -> dict[str, Any]:
    """Run a claim through the agent and return the decision envelope."""
    agent = _get_agent()
    try:
        result = agent.run(req.claim)
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.model_dump()


if __name__ == "__main__":
    # Allow `python -m shieldpoint_agents.api` to start the server.
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        log_level=os.environ.get("SHIELDPOINT_LOG_LEVEL", "info").lower(),
    )
