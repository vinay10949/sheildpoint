"""
LM Studio OpenAI-compatible client factory.

Wraps the OpenAI Python SDK so the rest of the framework doesn't hard-code
LM Studio URL/key handling. Centralizing this here lets us:

- Swap the underlying client (real OpenAI SDK vs. a test mock) without
  touching the Agent class.
- Enforce the timeout configured in :class:`AgentConfig`.
- Make the agent container reach LM Studio via ``host.docker.internal:1234``
  inside Docker while still allowing localhost in dev.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from .config import AgentConfig

logger = logging.getLogger("shieldpoint_agents.lmstudio")


class ChatCompleter(Protocol):
    """Minimal interface the Agent relies on.

    Any object exposing ``chat.completions.create(...)`` works — that includes
    the real OpenAI Python SDK client and any test double.
    """

    def chat(self) -> Any:  # pragma: no cover - structural typing
        ...


def build_lm_studio_client(
    config: AgentConfig,
    *,
    timeout: float | None = None,
) -> Any:
    """Build and return an OpenAI-compatible client pointed at LM Studio.

    The returned object is an ``openai.OpenAI`` instance. Tests should pass
    a fake client to :meth:`Agent.__init__` instead of calling this.
    """
    # Local import so the package can be imported even if openai isn't
    # installed yet (e.g. during bootstrap or doc build).
    from openai import OpenAI

    effective_timeout = timeout if timeout is not None else config.llm_timeout_sec
    logger.debug(
        "Building LM Studio client: base_url=%s model=%s timeout=%ss",
        config.lm_studio_base_url,
        config.model,
        effective_timeout,
    )
    return OpenAI(
        base_url=config.lm_studio_base_url,
        api_key=config.lm_studio_api_key,
        timeout=effective_timeout,
    )


def ping(client: Any, *, timeout: float = 2.0) -> bool:
    """Quick health probe — returns True iff LM Studio responds to ``/v1/models``.

    Used by :class:`Agent` to decide upfront whether to attempt the LLM call
    or skip straight to the fallback engine.
    """
    try:
        models = client.models.list()
        return bool(getattr(models, "data", None))
    except Exception as exc:
        logger.debug("LM Studio ping failed: %s", exc)
        return False
