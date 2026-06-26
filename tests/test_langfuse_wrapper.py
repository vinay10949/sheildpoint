"""
Unit tests for the ShieldPoint Langfuse tracer wrapper.

Run with:
    pip install pytest
    pytest tests/test_langfuse_wrapper.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Make agent_framework importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip Langfuse env vars before each test so we test from a clean state."""
    for k in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
              "LANGFUSE_ENABLED", "LANGFUSE_FLUSH_AT", "LANGFUSE_FLUSH_INTERVAL_MS"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Fixture: tracer instance with env vars reset
# ---------------------------------------------------------------------------
@pytest.fixture
def tracer(monkeypatch):
    """Yield the singleton tracer with env vars set for testing."""
    from agent_framework.observability.langfuse_wrapper import ShieldPointTracer
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test1234")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test1234")
    t = ShieldPointTracer()
    t.refresh()
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestShieldPointTracerLifecycle:
    def test_singleton_pattern(self, tracer):
        from agent_framework.observability.langfuse_wrapper import ShieldPointTracer
        assert ShieldPointTracer() is tracer

    def test_enabled_with_env_vars(self, tracer):
        assert tracer.enabled is True
        assert tracer.host == "http://localhost:3000"
        assert tracer.disabled_reason is None

    def test_disabled_without_env_vars(self, monkeypatch):
        from agent_framework.observability.langfuse_wrapper import ShieldPointTracer
        # env vars already stripped by clean_env fixture
        t = ShieldPointTracer()
        t.refresh()
        assert t.enabled is False
        assert "LANGFUSE_PUBLIC_KEY" in (t.disabled_reason or "")

    def test_disabled_via_env_var(self, monkeypatch):
        from agent_framework.observability.langfuse_wrapper import ShieldPointTracer
        monkeypatch.setenv("LANGFUSE_ENABLED", "false")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        t = ShieldPointTracer()
        t.refresh()
        assert t.enabled is False
        assert t.disabled_reason == "LANGFUSE_ENABLED=false"


class TestUsageExtraction:
    def test_extract_usage_from_openai_object(self):
        from agent_framework.observability.langfuse_wrapper import _extract_usage

        class _Usage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class _Msg:
            content = "hi"
            tool_calls = None

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()

        result = _extract_usage(_Resp())
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_extract_usage_from_dict(self):
        from agent_framework.observability.langfuse_wrapper import _extract_usage
        result = _extract_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        assert result == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_extract_usage_returns_none_for_no_usage(self):
        from agent_framework.observability.langfuse_wrapper import _extract_usage
        assert _extract_usage(None) is None
        assert _extract_usage("just a string") is None
        assert _extract_usage({}) is None

    def test_usage_to_v3_conversion(self):
        from agent_framework.observability.langfuse_wrapper import _usage_to_v3
        v2 = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        v3 = _usage_to_v3(v2)
        assert v3 == {"input": 100, "output": 50, "total": 150}
        assert _usage_to_v3(None) is None


class TestOutputExtraction:
    def test_extract_output_content(self):
        from agent_framework.observability.langfuse_wrapper import _extract_output_content

        class _Msg:
            content = "classification result"
            tool_calls = None

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]

        out = _extract_output_content(_Resp())
        assert out["content"] == "classification result"
        assert out["finish_reason"] == "stop"

    def test_extract_output_with_tool_calls(self):
        from agent_framework.observability.langfuse_wrapper import _extract_output_content

        class _Fn:
            name = "policy_validate"
            arguments = '{"policy_id": "P-001"}'

        class _TC:
            id = "call_001"
            type = "function"
            function = _Fn()

        class _Msg:
            content = None
            tool_calls = [_TC()]

        class _Choice:
            message = _Msg()
            finish_reason = "tool_calls"

        class _Resp:
            choices = [_Choice()]

        out = _extract_output_content(_Resp())
        assert out["tool_calls"][0]["function"]["name"] == "policy_validate"
        assert out["tool_calls"][0]["function"]["arguments"] == '{"policy_id": "P-001"}'


class TestDecorators:
    def test_observe_llm_decorator_runs_function(self, tracer):
        @tracer.observe_llm(name="test_fn")
        def my_fn(x):
            return x * 2

        # Even though Langfuse server isn't running, the wrapper should
        # not block the function call — it tries to send the trace async
        # and silently fails.
        result = my_fn(21)
        assert result == 42

    def test_observe_tool_decorator_runs_function(self, tracer):
        @tracer.observe_tool(name="test_tool")
        def my_tool(x):
            return f"result-{x}"

        result = my_tool("input")
        assert result == "result-input"

    def test_decorator_no_op_when_disabled(self, monkeypatch):
        from agent_framework.observability.langfuse_wrapper import ShieldPointTracer
        # No env vars → disabled
        t = ShieldPointTracer()
        t.refresh()
        assert not t.enabled

        @t.observe_llm(name="should_noop")
        def fn(x):
            return x + 1

        assert fn(5) == 6

    def test_decorator_preserves_function_metadata(self, tracer):
        @tracer.observe_llm(name="meta_test")
        def documented_function(a, b):
            """This is a docstring."""
            return a + b

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a docstring."

    def test_decorator_propagates_exceptions(self, tracer):
        @tracer.observe_llm(name="raises")
        def raises_fn():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            raises_fn()


class TestTraceContext:
    def test_trace_context_yields_handle(self, tracer):
        with tracer.trace(name="test_trace") as handle:
            assert handle.name == "test_trace"
            assert handle.active

    def test_trace_context_closes_handle(self, tracer):
        with tracer.trace(name="test_trace") as handle:
            pass
        assert not handle.active

    def test_trace_context_captures_exception(self, tracer):
        with pytest.raises(RuntimeError):
            with tracer.trace(name="error_trace") as handle:
                raise RuntimeError("boom")
        assert not handle.active


class TestConfig:
    def test_config_loads_defaults(self, monkeypatch):
        from agent_framework.config import load_config
        monkeypatch.setenv("LANGFUSE_HOST", "http://test:3000")
        cfg = load_config()
        assert cfg.langfuse.host == "http://test:3000"
        assert cfg.lm_studio.model == "qwen3.6-35b-a3b-q4_k_m"
        assert cfg.langfuse.retention_days == 90

    def test_config_respects_env_overrides(self, monkeypatch):
        from agent_framework.config import load_config
        monkeypatch.setenv("LANGFUSE_RETENTION_DAYS", "30")
        monkeypatch.setenv("QWEN_MODEL_ID", "qwen-custom")
        cfg = load_config()
        assert cfg.langfuse.retention_days == 30
        assert cfg.lm_studio.model == "qwen-custom"
