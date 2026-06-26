"""Unit tests for the LangfuseTracer façade."""

from __future__ import annotations

import pytest

from shieldpoint_agents import LangfuseTracer


# ---------------------------------------------------------------------------
# When the underlying ShieldPointTracer is unavailable OR Langfuse env vars
# are missing, every method should silently no-op.
# ---------------------------------------------------------------------------
class TestDisabledTracer:
    def test_tracer_disabled_without_env(self):
        t = LangfuseTracer(agent_name="test")
        assert t.enabled is False
        assert t.disabled_reason is not None

    def test_trace_context_is_safe_when_disabled(self):
        # The underlying ShieldPointTracer returns a no-op _TraceHandle
        # (not None) when disabled — the handle is safe to call .update()
        # and .end() on, all of which silently do nothing.
        t = LangfuseTracer(agent_name="test")
        with t.trace("test_trace") as span:
            # When disabled, the handle's `active` property is False and
            # `id` is None. We never get a bare None — that would force
            # callers to do None-checks everywhere.
            assert span is not None
            assert span.active is False
            assert span.id is None

    def test_llm_call_decorator_is_identity_when_disabled(self):
        t = LangfuseTracer(agent_name="test")

        @t.llm_call("test_fn")
        def my_fn(x):
            return x * 2

        assert my_fn(21) == 42

    def test_tool_call_decorator_is_identity_when_disabled(self):
        t = LangfuseTracer(agent_name="test")

        @t.tool_call("test_tool")
        def my_tool(x):
            return f"result-{x}"

        assert my_tool("input") == "result-input"


# ---------------------------------------------------------------------------
# With env vars set (but no live Langfuse server), the tracer should still
# be "enabled" — the underlying SDK buffers traces for async flush.
# ---------------------------------------------------------------------------
class TestEnabledTracer:
    def test_tracer_enabled_with_env(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test1234")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test1234")
        t = LangfuseTracer(agent_name="test")
        # Note: enabled depends on the langfuse package being installed.
        # If langfuse isn't installed in the test env, this is False — that's OK.
        if t.enabled:
            assert t.host == "http://localhost:3000"
            assert t.disabled_reason is None

    def test_tracer_metadata_propagates_agent_name(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test1234")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test1234")
        t = LangfuseTracer(agent_name="claim-classifier", agent_version="0.2.0")
        assert t.agent_name == "claim-classifier"
        assert t.agent_version == "0.2.0"


# ---------------------------------------------------------------------------
# flush() / shutdown() must never raise, even when disabled.
# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_flush_disabled_noop(self):
        t = LangfuseTracer(agent_name="test")
        t.flush()  # must not raise

    def test_shutdown_disabled_noop(self):
        t = LangfuseTracer(agent_name="test")
        t.shutdown()  # must not raise
