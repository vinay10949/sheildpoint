"""
LangfuseTracer — per-instance façade over the existing ShieldPointTracer.

The repo already ships ``agent_framework.observability.ShieldPointTracer``,
a singleton wrapping the Langfuse v3 Python SDK with decorators
(``observe_llm``, ``observe_tool``) and a ``trace()`` context manager. That
class is intentionally a singleton because the Langfuse SDK uses
OpenTelemetry context vars that need a single shared client to propagate
spans correctly.

The Agent framework, however, wants a *per-instance* tracer class so each
agent can carry its own metadata (name, version) without polluting a global.
This module reconciles both needs:

- :class:`LangfuseTracer` is a regular class (instantiable per-agent).
- It delegates all SDK work to the shared ``ShieldPointTracer`` singleton.
- It tracks per-instance metadata (agent name, claim id) and merges it into
  every trace it opens.
- If the Langfuse SDK isn't installed or env vars aren't set, every method
  silently no-ops — the agent runs untraced. This is the same fail-safe
  behaviour as the underlying wrapper.

API
---

::

    tracer = LangfuseTracer(agent_name="claim-classifier")

    # Open a top-level trace around an agent run
    with tracer.trace("agent_run", user_id="adjuster-42",
                      metadata={"claim_id": "CLM-2026-0001"}) as span:
        ...

    # Decorate a function that wraps an LLM call
    @tracer.llm_call("react_think")
    def call_llm(messages): ...

    # Decorate a tool invocation
    @tracer.tool_call("validate_policy")
    def validate_policy(policy_id): ...
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional, TypeVar

from .config import AgentConfig

logger = logging.getLogger("shieldpoint_agents.tracer")

F = TypeVar("F", bound=Callable[..., Any])


class LangfuseTracer:
    """Per-agent Langfuse tracer.

    Wraps the legacy ``ShieldPointTracer`` singleton. All SDK interactions
    go through the singleton; this class only adds per-instance metadata
    and a clean class-based API for new agent code.

    If the legacy tracer cannot be imported (e.g. ``agent_framework`` is
    not on the path), every method silently no-ops. Same if Langfuse env
    vars are missing.
    """

    def __init__(
        self,
        *,
        agent_name: str = "shieldpoint-agent",
        agent_version: str = "0.1.0",
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_version = agent_version
        self._config = config or AgentConfig.from_env()
        self._delegate = self._load_delegate()

    # ------------------------------------------------------------------ #
    #  Delegate loader — best-effort, never raises.                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_delegate() -> Optional[Any]:
        try:
            from agent_framework.observability.langfuse_wrapper import (
                ShieldPointTracer,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "ShieldPointTracer not importable; tracing disabled. (%s)", exc
            )
            return None

        try:
            tracer = ShieldPointTracer()
            tracer.refresh()
            return tracer
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize ShieldPointTracer: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Properties                                                         #
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        """True iff traces are actually being sent to Langfuse."""
        return bool(self._delegate and self._delegate.enabled)

    @property
    def host(self) -> Optional[str]:
        if self._delegate is None:
            return None
        return self._delegate.host

    @property
    def disabled_reason(self) -> Optional[str]:
        if self._delegate is None:
            return "agent_framework.observability.ShieldPointTracer not importable"
        return self._delegate.disabled_reason

    # ------------------------------------------------------------------ #
    #  Trace lifecycle                                                    #
    # ------------------------------------------------------------------ #
    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Iterator[Any]:
        """Open a top-level trace.

        Yields the underlying ``_TraceHandle`` from the legacy wrapper
        (or ``None`` if tracing is disabled). All metadata is merged with
        per-instance defaults (agent name, agent version).
        """
        merged_meta: dict[str, Any] = {
            "agent.name": self.agent_name,
            "agent.version": self.agent_version,
        }
        if metadata:
            merged_meta.update(metadata)

        merged_tags = list(tags or [])
        if self.agent_name not in merged_tags:
            merged_tags.append(self.agent_name)

        if self._delegate is None:
            # No-op context manager
            yield None
            return

        with self._delegate.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=merged_meta,
            tags=merged_tags,
            release=self.agent_version,
        ) as handle:
            yield handle

    # ------------------------------------------------------------------ #
    #  Decorators                                                         #
    # ------------------------------------------------------------------ #
    def llm_call(self, name: str) -> Callable[[F], F]:
        """Decorator for functions that wrap an LLM call.

        Captures: input prompt, model response, wall-clock latency, token
        counts (extracted from the OpenAI-style ``response.usage``).
        """
        if self._delegate is None or not self._delegate.enabled:
            return _identity_decorator

        return self._delegate.observe_llm(name=name)

    def tool_call(self, name: str) -> Callable[[F], F]:
        """Decorator for tool invocations registered in :class:`ToolRegistry`."""
        if self._delegate is None or not self._delegate.enabled:
            return _identity_decorator

        return self._delegate.observe_tool(name=name)

    # ------------------------------------------------------------------ #
    #  Flush / shutdown — pass-through to the singleton                   #
    # ------------------------------------------------------------------ #
    def flush(self) -> None:
        """Force-flush any buffered traces to Langfuse."""
        if self._delegate is None:
            return
        client = self._delegate.client
        if client is not None:
            try:
                client.flush()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Langfuse flush() failed: %s", exc)

    def shutdown(self) -> None:
        """Flush and release SDK resources. Call on agent process exit."""
        if self._delegate is None:
            return
        client = self._delegate.client
        if client is not None:
            try:
                # v3 SDK uses shutdown() to flush + close background workers.
                client.flush()
                if hasattr(client, "shutdown"):
                    client.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Langfuse shutdown() failed: %s", exc)


# ---------------------------------------------------------------------------
# Identity decorator — used when tracing is disabled so the wrapped function
# is returned unchanged (zero runtime overhead).
# ---------------------------------------------------------------------------
def _identity_decorator(func: F) -> F:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
