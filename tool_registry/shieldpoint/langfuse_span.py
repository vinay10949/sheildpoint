"""
Langfuse span recorder for tool invocations.

The real Langfuse Python SDK (``langfuse`` package) talks to a hosted or
self-hosted Langfuse server over HTTP. In production this is the desired
behaviour — every tool call appears as a span in the Langfuse UI with full
input/output, latency, and error context.

In tests and local development, however, we want to:

1. Capture spans in-memory so we can assert on them.
2. Avoid requiring a live Langfuse server (or even the ``langfuse`` package).

This module solves both with a small protocol, :class:`SpanRecorder`, and two
implementations:

- :class:`LangfuseSpanRecorder` — wraps the real Langfuse SDK. Falls back to
  no-op behaviour if the SDK or env vars are missing.
- :class:`NullSpanRecorder` — no-op, but keeps a list of recorded spans for
  test assertions.

The :class:`ToolRegistry` only ever talks to the protocol, so swapping
implementations is trivial.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("shieldpoint.langfuse_span")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class SpanRecorder(Protocol):
    """Protocol every span recorder must implement.

    The registry calls :meth:`start_tool_span` before invoking a tool and
    :meth:`end_tool_span` after it returns (or raises). Failed validation
    attempts (which never reach the function) go through
    :meth:`record_failed_tool_call` instead.
    """

    def start_tool_span(self, *, name: str, input: dict[str, Any]) -> Any:
        """Open a span around a tool invocation. Returns an opaque handle."""
        ...

    def end_tool_span(
        self,
        *,
        handle: Any,
        name: str,
        input: dict[str, Any],
        output: Any,
        error: Optional[BaseException],
        latency_ms: float,
    ) -> None:
        """Close a previously opened span, recording output / error / latency."""
        ...

    def record_failed_tool_call(
        self, *, name: str, input: dict[str, Any], error: BaseException
    ) -> None:
        """Record a tool call that failed schema validation (never invoked)."""
        ...


# ---------------------------------------------------------------------------
# Span dataclass (used by both NullSpanRecorder and tests)
# ---------------------------------------------------------------------------
@dataclass
class RecordedSpan:
    """A single recorded tool-call span."""

    name: str
    input: dict[str, Any]
    output: Any = None
    error: Optional[BaseException] = None
    latency_ms: float = 0.0
    status: str = "ok"  # one of: ok, error, validation_failed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for assertion / logging."""
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "error": repr(self.error) if self.error else None,
            "latency_ms": self.latency_ms,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# NullSpanRecorder — default; captures spans in-memory for assertions
# ---------------------------------------------------------------------------
class NullSpanRecorder:
    """No-op span recorder that keeps an in-memory audit log.

    This is the default recorder used by :class:`ToolRegistry`. It does not
    send anything to Langfuse, but it *does* retain every recorded span in
    ``self.spans`` so tests can assert on tool-call provenance.

    The list is append-only and protected by a lock so it is safe to use
    from multiple threads (e.g. when the agent runs tools concurrently).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.spans: list[RecordedSpan] = []

    def start_tool_span(self, *, name: str, input: dict[str, Any]) -> RecordedSpan:
        span = RecordedSpan(name=name, input=dict(input), status="ok")
        return span

    def end_tool_span(
        self,
        *,
        handle: RecordedSpan,
        name: str,
        input: dict[str, Any],
        output: Any,
        error: Optional[BaseException],
        latency_ms: float,
    ) -> None:
        handle.output = output
        handle.error = error
        handle.latency_ms = latency_ms
        handle.status = "error" if error is not None else "ok"
        with self._lock:
            self.spans.append(handle)

    def record_failed_tool_call(
        self, *, name: str, input: dict[str, Any], error: BaseException
    ) -> None:
        span = RecordedSpan(
            name=name,
            input=dict(input),
            error=error,
            status="validation_failed",
        )
        with self._lock:
            self.spans.append(span)

    # Convenience accessors for tests
    def spans_for(self, name: str) -> list[RecordedSpan]:
        return [s for s in self.spans if s.name == name]

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()


# ---------------------------------------------------------------------------
# LangfuseSpanRecorder — wraps the real Langfuse SDK (optional dep)
# ---------------------------------------------------------------------------
class LangfuseSpanRecorder:
    """Span recorder that ships spans to a live Langfuse server.

    Uses the ``langfuse`` v2/v3 Python SDK if it is importable AND the
    ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` env vars are set.
    Otherwise it silently degrades to no-op behaviour (same fail-safe as the
    repo's existing ``ShieldPointTracer``).

    The recorder is intentionally minimal: it opens one span per tool call
    via ``langfuse.span()``, sets I/O metadata, and ends the span on return.
    For richer tracing (nested spans, OTel context propagation) the agent
    loop should layer additional decorators on top.
    """

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        agent_name: str = "shieldpoint-agent",
        agent_version: str = "0.1.0",
    ) -> None:
        self.agent_name = agent_name
        self.agent_version = agent_version
        self._public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        self._secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        self._host = host or os.environ.get(
            "LANGFUSE_HOST", "http://localhost:3000"
        )
        self._client = self._init_client()
        # Always keep an in-memory mirror so tests / audits can read spans
        # even when Langfuse is unreachable.
        self._mirror = NullSpanRecorder()

    # ------------------------------------------------------------------ #
    #  Client init — best-effort, never raises.                          #
    # ------------------------------------------------------------------ #
    def _init_client(self) -> Optional[Any]:
        if not (self._public_key and self._secret_key):
            logger.debug(
                "Langfuse env vars missing; spans will be recorded locally only."
            )
            return None
        try:
            from langfuse import Langfuse  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("langfuse SDK not importable; tracing disabled. (%s)", exc)
            return None
        try:
            return Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
                release=self.agent_version,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialize Langfuse client: %s", exc)
            return None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ #
    #  SpanRecorder protocol                                             #
    # ------------------------------------------------------------------ #
    def start_tool_span(self, *, name: str, input: dict[str, Any]) -> Any:
        # Always record locally first.
        local = self._mirror.start_tool_span(name=name, input=input)

        if self._client is None:
            return local

        try:
            handle = self._client.span(
                name=f"tool.{name}",
                metadata={
                    "tool.name": name,
                    "tool.input": input,
                    "agent.name": self.agent_name,
                    "agent.version": self.agent_version,
                },
            )
            # Stash the local mirror on the handle so end_tool_span can update it.
            setattr(handle, "_local_mirror", local)
            return handle
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Langfuse span() failed: %s", exc)
            return local

    def end_tool_span(
        self,
        *,
        handle: Any,
        name: str,
        input: dict[str, Any],
        output: Any,
        error: Optional[BaseException],
        latency_ms: float,
    ) -> None:
        # Update local mirror first so it always reflects the final state.
        local = getattr(handle, "_local_mirror", handle)
        if isinstance(local, RecordedSpan):
            self._mirror.end_tool_span(
                handle=local,
                name=name,
                input=input,
                output=output,
                error=error,
                latency_ms=latency_ms,
            )

        if self._client is None or not isinstance(handle, RecordedSpan):
            # Either no client, or handle was already the local mirror.
            return
        try:
            # The real SDK handle (when present) is the object returned by
            # ``langfuse.span()``. It exposes ``end()`` to close the span.
            if hasattr(handle, "end"):
                handle.end(
                    output=output,
                    metadata={
                        "tool.output": output,
                        "tool.error": repr(error) if error else None,
                        "tool.latency_ms": latency_ms,
                        "tool.status": "error" if error else "ok",
                    },
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Langfuse span.end() failed: %s", exc)

    def record_failed_tool_call(
        self, *, name: str, input: dict[str, Any], error: BaseException
    ) -> None:
        self._mirror.record_failed_tool_call(name=name, input=input, error=error)
        if self._client is None:
            return
        try:
            span = self._client.span(
                name=f"tool.{name}.validation_failed",
                metadata={
                    "tool.name": name,
                    "tool.input": input,
                    "tool.error": repr(error),
                    "tool.status": "validation_failed",
                    "agent.name": self.agent_name,
                },
            )
            if hasattr(span, "end"):
                span.end(output=None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Langfuse validation-failed span failed: %s", exc)

    # ------------------------------------------------------------------ #
    #  Mirror access — lets tests read spans even when Langfuse is live. #
    # ------------------------------------------------------------------ #
    @property
    def spans(self) -> list[RecordedSpan]:
        return self._mirror.spans

    def spans_for(self, name: str) -> list[RecordedSpan]:
        return self._mirror.spans_for(name)

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Langfuse flush() failed: %s", exc)

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
                if hasattr(self._client, "shutdown"):
                    self._client.shutdown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Langfuse shutdown() failed: %s", exc)
