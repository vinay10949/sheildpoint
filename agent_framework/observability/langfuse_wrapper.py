"""
ShieldPoint Langfuse Tracer (Langfuse v3 SDK)
==============================================

Wrapper class around the Langfuse Python SDK (``langfuse>=2.0`` — installs
v3.x by default, which is API-compatible with both Langfuse v2.x and v3.x
self-hosted servers). Decorates LLM calls and tool invocations so every
agent run is traced end-to-end. Captures input prompts, model responses,
latency, token counts, and tool invocations to the self-hosted Langfuse
instance running on the ShieldPoint internal network.

Design goals
------------
1. **Zero-friction adoption**: decorating an LLM call is one line
   (``@observe_llm``). If the Langfuse SDK is not installed or the env vars
   are not set, the decorator transparently no-ops — agent code still runs.
2. **Network-boundary safe**: the wrapper ONLY talks to ``LANGFUSE_HOST``
   (defaults to ``http://localhost:3000``). No third-party endpoints.
3. **Rich capture**: extracts token usage from OpenAI-style responses,
   records tool calls (name, args, result, error), and tags each trace with
   the agent name + claim_id when available.
4. **Context propagation**: v3 uses OpenTelemetry context vars under the
   hood — spans created inside a ``trace_context()`` block are
   automatically attached to the parent trace, no manual plumbing needed.

v3 SDK API used
---------------
- ``Langfuse(public_key=, secret_key=, host=, flush_at=, flush_interval=)``
- ``lf.start_as_current_span(name=, input=, output=, metadata=)``
- ``lf.start_as_current_generation(name=, input=, output=, model=,
                                   usage_details=, metadata=)``
- ``lf.update_current_trace(user_id=, session_id=, metadata=, tags=,
                            release=, version=)``
- ``lf.get_current_trace_id()``
- ``lf.flush()`` / ``lf.shutdown()``

Usage
-----
Decorator style (recommended):

    from shieldpoint.observability import observe_llm, observe_tool, trace_context

    @observe_llm(name="classify_claim")
    def classify_claim(claim_text: str, model: str) -> dict:
        client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": claim_text}],
        )

    @observe_tool(name="policy_validate")
    def policy_validate(policy_id: str, claim: dict) -> dict:
        ...

    # Top-level agent run — creates the trace boundary
    with trace_context(name="agent_run", user_id="adjuster-42",
                       metadata={"claim_id": "CLM-2026-0001"}):
        result = classify_claim(claim_text, model="qwen3.6-35b-a3b-q4_k_m")
        policy_result = policy_validate(policy_id, claim)

Environment variables
---------------------
- ``LANGFUSE_HOST``         — base URL of the self-hosted Langfuse (default
                              ``http://localhost:3000``). Inside a container,
                              use ``http://langfuse:3000``.
- ``LANGFUSE_PUBLIC_KEY``   — project public key (``pk-lf-...``).
- ``LANGFUSE_SECRET_KEY``   — project secret key (``sk-lf-...``).
- ``LANGFUSE_FLUSH_AT``     — batch size for SDK flush (default 15).
- ``LANGFUSE_FLUSH_INTERVAL_MS`` — max time between flushes (default 1000ms).
- ``LANGFUSE_ENABLED``      — set to ``false`` to disable at runtime even if
                              keys are present (useful for tests).
"""

from __future__ import annotations

import functools
import logging
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar

# ---------------------------------------------------------------------------
# Optional Langfuse SDK import. If the package isn't installed (e.g. in CI
# without dev deps), the tracer degrades to a no-op so agent code still runs.
# ---------------------------------------------------------------------------
try:
    from langfuse import Langfuse
    _LANGFUSE_SDK_AVAILABLE = True
    _LANGFUSE_VERSION: Optional[str]
    try:
        from langfuse.version import __version__ as _lf_ver
        _LANGFUSE_VERSION = _lf_ver
    except Exception:
        _LANGFUSE_VERSION = "unknown"
except ImportError:  # pragma: no cover
    Langfuse = None  # type: ignore
    _LANGFUSE_SDK_AVAILABLE = False
    _LANGFUSE_VERSION = None


logger = logging.getLogger("shieldpoint.observability")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LangfuseNotConfiguredError(RuntimeError):
    """Raised when caller explicitly requires tracing but it is not active."""


# ---------------------------------------------------------------------------
# Usage extraction — pulls prompt/completion/total token counts out of an
# OpenAI-style chat completion response (LM Studio returns this shape).
# Returns a dict in the v3 ``usage_details`` format:
#   {"input": int, "output": int, "total": int}
# (v3 uses these keys instead of v2's prompt_tokens/completion_tokens).
# ---------------------------------------------------------------------------
def _extract_usage(result: Any) -> Optional[Dict[str, int]]:
    """Extract token usage from an OpenAI-style response object.

    Returns a dict with keys ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens`` (v2-style names — converted to v3 ``usage_details``
    keys ``input`` / ``output`` / ``total`` at trace-attach time).
    """
    if result is None:
        return None

    # OpenAI Python SDK v1+: response.usage is a CompletionUsage object
    usage = getattr(result, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        if prompt is not None and completion is not None:
            return {
                "prompt_tokens": int(prompt),
                "completion_tokens": int(completion),
                "total_tokens": int(total) if total is not None
                    else int(prompt) + int(completion),
            }

    # Dict form (raw JSON response)
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if prompt is not None and completion is not None:
            total = usage.get("total_tokens", int(prompt) + int(completion))
            return {
                "prompt_tokens": int(prompt),
                "completion_tokens": int(completion),
                "total_tokens": int(total),
            }

    # Some clients return the full response as a dict
    if isinstance(result, dict):
        usage = result.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if prompt is not None and completion is not None:
                total = usage.get("total_tokens", int(prompt) + int(completion))
                return {
                    "prompt_tokens": int(prompt),
                    "completion_tokens": int(completion),
                    "total_tokens": int(total),
                }

    return None


def _usage_to_v3(usage: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """Convert v2-style usage dict to v3 ``usage_details`` format."""
    if usage is None:
        return None
    return {
        "input": usage.get("prompt_tokens", 0),
        "output": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens",
                           usage.get("prompt_tokens", 0)
                           + usage.get("completion_tokens", 0)),
    }


def _extract_output_content(result: Any) -> Any:
    """Extract a serializable representation of the LLM response."""
    if result is None:
        return None
    # OpenAI SDK v1+: response.choices[0].message.content
    choices = getattr(result, "choices", None)
    if choices and len(choices) > 0:
        msg = getattr(choices[0], "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
            output: Dict[str, Any] = {}
            if content is not None:
                output["content"] = content
            if tool_calls:
                # Capture tool invocations the LLM requested
                tc_list = []
                for tc in tool_calls:
                    tc_list.append({
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", None),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                        },
                    })
                output["tool_calls"] = tc_list
            output["finish_reason"] = getattr(choices[0], "finish_reason", None)
            return output
    # Dict-style response
    if isinstance(result, dict):
        return result
    # Fallback: stringify
    try:
        return str(result)
    except Exception:
        return "<unserializable>"


def _safe_serialize(value: Any) -> Any:
    """Best-effort JSON-friendly serialization of any value."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_serialize(v) for v in value]
    # Objects with __dict__ (Pydantic, dataclasses, OpenAI SDK objects)
    if hasattr(value, "model_dump"):  # Pydantic v2
        try:
            return _safe_serialize(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: _safe_serialize(v) for k, v in vars(value).items()
                if not k.startswith("_") and _is_jsonable(v)}
    return str(value)


def _is_jsonable(value: Any) -> bool:
    try:
        import json
        json.dumps(value)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Trace handle — opaque handle returned by ``tracer.trace(...)`` /
# ``trace_context(...)``. Wraps the v3 OpenTelemetry span so callers can
# attach metadata or end the trace explicitly.
# ---------------------------------------------------------------------------
class _TraceHandle:
    """Opaque handle for a trace.

    Wraps the v3 ``LangfuseSpan`` (or None when tracing is disabled). Use
    ``.update()``, ``.end()`` to interact. All methods are safe no-ops when
    tracing is disabled.
    """

    __slots__ = ("_client", "_span", "_ended", "_name")

    def __init__(
        self,
        client: Optional[Any],
        span: Optional[Any],
        name: str,
    ) -> None:
        self._client = client
        self._span = span
        self._name = name
        self._ended = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def active(self) -> bool:
        return self._span is not None and not self._ended

    @property
    def id(self) -> Optional[str]:
        """Return the trace ID (not the span ID) — useful for verification."""
        if not self.active or self._client is None:
            return None
        try:
            return self._client.get_current_trace_id()
        except Exception:
            return None

    def update(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Update trace-level fields."""
        if not self.active:
            return
        try:
            kwargs: Dict[str, Any] = {}
            if user_id is not None: kwargs["user_id"] = user_id
            if session_id is not None: kwargs["session_id"] = session_id
            if metadata is not None: kwargs["metadata"] = metadata
            if tags is not None: kwargs["tags"] = tags
            if release is not None: kwargs["release"] = release
            if version is not None: kwargs["version"] = version
            if kwargs:
                self._client.update_current_trace(**kwargs)
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse update_current_trace() failed: %s", exc)

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        if self._span is not None:
            try:
                self._span.end()
            except Exception as exc:  # pragma: no cover
                logger.debug("Langfuse span.end() failed: %s", exc)
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:  # pragma: no cover
                logger.debug("Langfuse flush() failed: %s", exc)

    def __enter__(self) -> "_TraceHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None and self.active:
            try:
                self._span.update(
                    level="ERROR",
                    status_message=f"{exc_type.__name__ if exc_type else 'Unknown'}: {exc_val}",
                    metadata={
                        "error_type": exc_type.__name__ if exc_type else "Unknown",
                        "error_message": str(exc_val),
                        "stack_trace": "".join(
                            traceback.format_exception(exc_type, exc_val, exc_tb)
                        ),
                    },
                )
            except Exception:
                pass
        self.end()


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------
class ShieldPointTracer:
    """Singleton tracer wrapping the Langfuse v3 Python SDK.

    Instantiate once at process start (``tracer = ShieldPointTracer()`` or
    just import ``tracer`` from this module). Reads configuration from env
    vars on first construction; call ``tracer.refresh()`` to re-read after
    env changes.
    """

    _instance: Optional["ShieldPointTracer"] = None

    def __new__(cls) -> "ShieldPointTracer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._client: Optional[Any] = None
        self._host: Optional[str] = None
        self._public_key: Optional[str] = None
        self._enabled: bool = False
        self._disabled_reason: Optional[str] = None
        self.refresh()

    # ---- lifecycle ----------------------------------------------------------
    def refresh(self) -> None:
        """Re-read env vars and re-initialize the underlying Langfuse client."""
        enabled_flag = os.environ.get("LANGFUSE_ENABLED", "true").lower()
        if enabled_flag in ("0", "false", "no", "off"):
            self._enabled = False
            self._disabled_reason = "LANGFUSE_ENABLED=false"
            self._client = None
            logger.info("Langfuse tracing disabled via LANGFUSE_ENABLED env var.")
            return

        if not _LANGFUSE_SDK_AVAILABLE:
            self._enabled = False
            self._disabled_reason = (
                "langfuse package not installed — pip install 'langfuse>=2.0'"
            )
            self._client = None
            logger.warning(
                "Langfuse SDK not installed — tracing disabled. "
                "Install with: pip install 'langfuse>=2.0'"
            )
            return

        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            self._enabled = False
            self._disabled_reason = (
                "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set in env"
            )
            self._client = None
            logger.warning(
                "Langfuse public/secret keys not set — tracing disabled. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars."
            )
            return

        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
                flush_at=int(os.environ.get("LANGFUSE_FLUSH_AT", "15")),
                flush_interval=float(
                    os.environ.get("LANGFUSE_FLUSH_INTERVAL_MS", "1000")
                ),
            )
            self._host = host
            self._public_key = public_key
            self._enabled = True
            self._disabled_reason = None
            logger.info(
                "Langfuse tracing enabled → %s (SDK v%s)",
                host, _LANGFUSE_VERSION,
            )
        except Exception as exc:
            self._enabled = False
            self._disabled_reason = f"Langfuse client init failed: {exc}"
            self._client = None
            logger.error("Failed to initialize Langfuse client: %s", exc)

    # ---- properties ---------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True iff traces are being captured and sent to Langfuse."""
        return self._enabled and self._client is not None

    @property
    def host(self) -> Optional[str]:
        return self._host

    @property
    def disabled_reason(self) -> Optional[str]:
        return self._disabled_reason

    @property
    def client(self) -> Optional[Any]:
        """Direct access to the underlying Langfuse client (or None)."""
        return self._client

    # ---- trace creation -----------------------------------------------------
    def trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        release: Optional[str] = None,
        version: Optional[str] = None,
    ) -> _TraceHandle:
        """Start a new trace (top-level span). Use as a context manager
        or call ``.end()``.

        If tracing is disabled, returns a no-op handle (safe to call
        ``.update()``, ``.end()`` on it — all silently do nothing).
        """
        if not self.enabled or self._client is None:
            return _TraceHandle(None, None, name)

        try:
            span = self._client.start_as_current_span(
                name=name,
                metadata=metadata or {},
            )
            span.__enter__()
            # Update trace-level fields (v3 separates span from trace metadata)
            update_kwargs: Dict[str, Any] = {}
            if user_id is not None: update_kwargs["user_id"] = user_id
            if session_id is not None: update_kwargs["session_id"] = session_id
            if tags is not None: update_kwargs["tags"] = tags
            if release is not None: update_kwargs["release"] = release
            if version is not None: update_kwargs["version"] = version
            if metadata is not None:
                update_kwargs["metadata"] = metadata
            if update_kwargs:
                try:
                    self._client.update_current_trace(**update_kwargs)
                except Exception as exc:  # pragma: no cover
                    logger.debug("update_current_trace failed: %s", exc)
        except Exception as exc:
            logger.debug("Langfuse start_as_current_span failed: %s", exc)
            return _TraceHandle(None, None, name)

        return _TraceHandle(self._client, span, name)

    # ---- decorators ---------------------------------------------------------
    def observe_llm(
        self,
        name: Optional[str] = None,
        model_env_var: str = "QWEN_MODEL_ID",
    ) -> Callable[[F], F]:
        """Decorator: wrap an LLM-calling function to capture a generation.

        Captures: input args, output (response content + tool_calls),
        latency_ms, token usage (extracted from response.usage), errors.

        The wrapped function MUST return either:
        - An OpenAI SDK ChatCompletion object (preferred), OR
        - A dict with the same shape (response.usage.prompt_tokens etc.), OR
        - Any other value (no token usage extracted, output is stringified).

        The function's first positional arg OR ``messages`` kwarg is logged
        as the prompt when present.
        """
        def decorator(fn: F) -> F:
            generation_name = name or fn.__name__
            default_model = os.environ.get(model_env_var, "unknown")

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled or self._client is None:
                    return fn(*args, **kwargs)

                model = kwargs.get("model", default_model)
                input_repr = _build_llm_input_repr(args, kwargs)

                start_perf = time.perf_counter()
                error: Optional[BaseException] = None
                result: Any = None
                # Open a generation span — auto-attaches to current trace
                # if one is open, else creates a new trace implicitly.
                # NOTE: ``start_as_current_generation`` is deprecated in v3.x
                # in favor of ``start_as_current_observation(as_type='generation')``.
                # We try the new API first, fall back to the old one for v2.x.
                try:
                    if hasattr(self._client, "start_as_current_observation"):
                        gen_cm = self._client.start_as_current_observation(
                            as_type="generation",
                            name=generation_name,
                            input=input_repr,
                            model=str(model),
                        )
                    else:
                        gen_cm = self._client.start_as_current_generation(
                            name=generation_name,
                            input=input_repr,
                            model=str(model),
                        )
                    gen_span = gen_cm.__enter__()
                except Exception as exc:
                    logger.debug("start_as_current_observation/generation failed: %s", exc)
                    # Fall back: just call the function untraced
                    return fn(*args, **kwargs)

                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    error = exc
                    raise
                finally:
                    latency_ms = int((time.perf_counter() - start_perf) * 1000)
                    metadata: Dict[str, Any] = {
                        "function": fn.__qualname__,
                        "module": fn.__module__,
                        "latency_ms": latency_ms,
                    }
                    if error is not None:
                        metadata["error"] = f"{type(error).__name__}: {error}"
                        metadata["stack_trace"] = traceback.format_exc()

                    usage = _extract_usage(result) if error is None else None
                    usage_v3 = _usage_to_v3(usage)
                    output = _extract_output_content(result) if error is None else None

                    try:
                        update_kwargs: Dict[str, Any] = {
                            "output": _safe_serialize(output),
                            "metadata": metadata,
                        }
                        if usage_v3 is not None:
                            update_kwargs["usage_details"] = usage_v3
                        if error is not None:
                            update_kwargs["level"] = "ERROR"
                            update_kwargs["status_message"] = str(error)
                        gen_span.update(**update_kwargs)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("gen.update failed: %s", exc)
                    try:
                        gen_cm.__exit__(
                            type(error) if error else None,
                            error,
                            traceback.extract_tb(error.__traceback__)
                                if error else None,
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.debug("gen.__exit__ failed: %s", exc)

            return wrapper  # type: ignore[return-value]

        return decorator

    def observe_tool(
        self,
        name: Optional[str] = None,
    ) -> Callable[[F], F]:
        """Decorator: wrap a tool function to capture a span.

        Captures: tool name, args, result (or error), latency.
        Tool invocations requested by the LLM (function calling) should be
        wrapped with this so they appear as spans on the trace.
        """
        def decorator(fn: F) -> F:
            tool_name = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if not self.enabled or self._client is None:
                    return fn(*args, **kwargs)

                start_perf = time.perf_counter()
                error: Optional[BaseException] = None
                result: Any = None
                input_repr = _build_tool_input_repr(args, kwargs)

                try:
                    span_cm = self._client.start_as_current_span(
                        name=tool_name,
                        input=input_repr,
                    )
                    span = span_cm.__enter__()
                except Exception as exc:
                    logger.debug("start_as_current_span failed: %s", exc)
                    return fn(*args, **kwargs)

                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    error = exc
                    raise
                finally:
                    latency_ms = int((time.perf_counter() - start_perf) * 1000)
                    metadata: Dict[str, Any] = {
                        "function": fn.__qualname__,
                        "module": fn.__module__,
                        "latency_ms": latency_ms,
                    }
                    if error is not None:
                        metadata["error"] = f"{type(error).__name__}: {error}"
                        metadata["stack_trace"] = traceback.format_exc()
                    try:
                        update_kwargs: Dict[str, Any] = {
                            "output": _safe_serialize(result) if error is None else None,
                            "metadata": metadata,
                        }
                        if error is not None:
                            update_kwargs["level"] = "ERROR"
                            update_kwargs["status_message"] = str(error)
                        span.update(**update_kwargs)
                    except Exception as exc:  # pragma: no cover
                        logger.debug("span.update failed: %s", exc)
                    try:
                        span_cm.__exit__(
                            type(error) if error else None,
                            error,
                            traceback.extract_tb(error.__traceback__)
                                if error else None,
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.debug("span.__exit__ failed: %s", exc)

            return wrapper  # type: ignore[return-value]

        return decorator

    # ---- explicit flush / shutdown -----------------------------------------
    def flush(self) -> None:
        """Flush any pending trace events to Langfuse immediately."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:  # pragma: no cover
                logger.debug("Langfuse flush failed: %s", exc)

    def shutdown(self) -> None:
        """Flush + close. Call at process exit to avoid losing buffered events."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers used by the decorators
# ---------------------------------------------------------------------------
def _build_llm_input_repr(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Build a JSON-friendly representation of the LLM call input."""
    repr_: Dict[str, Any] = {}
    if "messages" in kwargs:
        repr_["messages"] = _safe_serialize(kwargs["messages"])
    elif args and isinstance(args[0], list):
        repr_["messages"] = _safe_serialize(args[0])
    if "model" in kwargs:
        repr_["model"] = kwargs["model"]
    if "temperature" in kwargs:
        repr_["temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs:
        repr_["max_tokens"] = kwargs["max_tokens"]
    if "tools" in kwargs:
        repr_["tools"] = _safe_serialize(kwargs["tools"])
    if "tool_choice" in kwargs:
        repr_["tool_choice"] = kwargs["tool_choice"]
    other = {k: _safe_serialize(v) for k, v in kwargs.items()
             if k not in {"model", "messages", "temperature", "max_tokens",
                          "tools", "tool_choice"}}
    if other:
        repr_["params"] = other
    return repr_


def _build_tool_input_repr(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Build a JSON-friendly representation of the tool invocation input."""
    repr_: Dict[str, Any] = {"args": _safe_serialize(args)}
    if kwargs:
        repr_["kwargs"] = _safe_serialize(kwargs)
    return repr_


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere.
# ---------------------------------------------------------------------------
tracer = ShieldPointTracer()


# ---------------------------------------------------------------------------
# Convenience module-level decorators — bound to the singleton tracer.
# ---------------------------------------------------------------------------
def observe_llm(name: Optional[str] = None, **kwargs: Any) -> Callable[[F], F]:
    """Module-level shortcut for ``tracer.observe_llm(name=..., **kwargs)``."""
    return tracer.observe_llm(name=name, **kwargs)


def observe_tool(name: Optional[str] = None) -> Callable[[F], F]:
    """Module-level shortcut for ``tracer.observe_tool(name=...)``."""
    return tracer.observe_tool(name=name)


@contextmanager
def trace_context(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
):
    """Context manager that opens a trace, yields the handle, then closes it.

    Example::

        with trace_context(name="agent_run", user_id="adjuster-42"):
            result = classify_claim(...)  # decorated with @observe_llm
            policy_result = policy_validate(...)  # decorated with @observe_tool
    """
    handle = tracer.trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        tags=tags,
    )
    try:
        yield handle
    except Exception as exc:
        try:
            handle.update(metadata={
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stack_trace": traceback.format_exc(),
            })
        except Exception:
            pass
        raise
    finally:
        handle.end()


def current_trace_id() -> Optional[str]:
    """Return the trace ID active in the current context, if any."""
    if not tracer.enabled:
        return None
    try:
        return tracer.client.get_current_trace_id()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# atexit hook — flush on interpreter exit so no traces are lost.
# ---------------------------------------------------------------------------
import atexit
atexit.register(tracer.shutdown)
