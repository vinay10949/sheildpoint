"""Tests for the ToolRegistry base class — register_tool / get_tool / invoke + schema validation."""
from __future__ import annotations

import pytest

from shieldpoint import (
    Tool,
    ToolInvocationError,
    ToolNotFoundError,
    ToolRegistry,
    ToolValidationError,
)
from shieldpoint.langfuse_span import NullSpanRecorder


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def add(x: int, y: int = 0) -> int:
    """Add two integers."""
    return x + y


def boom(**kwargs):
    """A tool that always raises."""
    raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# register_tool
# ---------------------------------------------------------------------------
class TestRegisterTool:
    def test_register_with_explicit_schema(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer", "default": 0},
                },
                "required": ["x"],
            },
        )
        assert "add" in reg
        assert reg.names() == ["add"]

    def test_register_as_decorator(self):
        reg = ToolRegistry()

        @reg.register_tool(
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        def add(x: int) -> int:
            return x

        # Decorator returns the original function unchanged.
        assert add(5) == 5
        assert "add" in reg

    def test_register_infers_schema_from_signature(self):
        reg = ToolRegistry()
        reg.register_tool(add, name="add", description="Add two integers.")
        tool = reg.get_tool("add")
        assert tool.parameters["properties"]["x"]["type"] == "integer"
        assert tool.parameters["properties"]["y"]["type"] == "integer"
        assert "x" in tool.parameters["required"]
        assert "y" not in tool.parameters.get("required", [])

    def test_register_requires_description(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="no description"):

            def no_doc(x):
                return x

            reg.register_tool(no_doc, name="no_doc")

    def test_register_returns_tool_on_direct_call(self):
        reg = ToolRegistry()
        tool = reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={"type": "object"},
        )
        assert isinstance(tool, Tool)
        assert tool.name == "add"

    def test_backwards_compat_register_alias(self):
        """The legacy `register` alias must still work."""
        reg = ToolRegistry()
        reg.register(
            add,
            name="add",
            description="Add two integers.",
            schema={"type": "object"},
        )
        assert "add" in reg


# ---------------------------------------------------------------------------
# get_tool
# ---------------------------------------------------------------------------
class TestGetTool:
    def test_get_tool_returns_tool(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={"type": "object"},
        )
        tool = reg.get_tool("add")
        assert isinstance(tool, Tool)
        assert tool.name == "add"

    def test_get_tool_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError, match="not registered"):
            reg.get_tool("nope")

    def test_backwards_compat_get_alias(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="...",
            schema={"type": "object"},
        )
        assert reg.get("add") is reg.get_tool("add")


# ---------------------------------------------------------------------------
# invoke + schema validation
# ---------------------------------------------------------------------------
class TestInvoke:
    def test_invoke_valid_args(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
            },
        )
        assert reg.invoke("add", x=3, y=4) == 7

    def test_invoke_rejects_wrong_type(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )
        with pytest.raises(ToolValidationError) as exc_info:
            reg.invoke("add", x="not-an-int")
        assert exc_info.value.tool == "add"
        assert exc_info.value.details["validator"] == "type"
        assert exc_info.value.details["json_path"] == "$.x"
        assert "input" in exc_info.value.details
        assert exc_info.value.details["input"] == {"x": "not-an-int"}

    def test_invoke_rejects_missing_required(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        )
        with pytest.raises(ToolValidationError, match="required"):
            reg.invoke("add", x=3)

    def test_invoke_rejects_additional_properties(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )
        with pytest.raises(ToolValidationError, match="Additional"):
            reg.invoke("add", x=3, surprise="!")

    def test_invoke_unknown_tool(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.invoke("nope")

    def test_invoke_wraps_function_exception(self):
        reg = ToolRegistry()
        reg.register_tool(
            boom,
            name="boom",
            description="Always raises.",
            schema={"type": "object"},
        )
        with pytest.raises(ToolInvocationError) as exc_info:
            reg.invoke("boom")
        assert exc_info.value.tool == "boom"
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "kaboom" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# Span recording
# ---------------------------------------------------------------------------
class TestSpanRecording:
    def test_successful_invoke_records_span(self, span_recorder):
        reg = ToolRegistry(span_recorder=span_recorder)
        reg.register_tool(
            add,
            name="add",
            description="...",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        reg.invoke("add", x=5)
        spans = span_recorder.spans_for("add")
        assert len(spans) == 1
        assert spans[0].status == "ok"
        assert spans[0].output == 5
        assert spans[0].error is None
        assert spans[0].latency_ms > 0

    def test_failed_validation_records_span(self, span_recorder):
        reg = ToolRegistry(span_recorder=span_recorder)
        reg.register_tool(
            add,
            name="add",
            description="...",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )
        with pytest.raises(ToolValidationError):
            reg.invoke("add", x="bad")
        spans = span_recorder.spans_for("add")
        assert len(spans) == 1
        assert spans[0].status == "validation_failed"
        assert spans[0].error is not None

    def test_function_exception_records_span(self, span_recorder):
        reg = ToolRegistry(span_recorder=span_recorder)
        reg.register_tool(
            boom,
            name="boom",
            description="...",
            schema={"type": "object"},
        )
        with pytest.raises(ToolInvocationError):
            reg.invoke("boom")
        spans = span_recorder.spans_for("boom")
        assert len(spans) == 1
        assert spans[0].status == "error"
        assert spans[0].error is not None


# ---------------------------------------------------------------------------
# OpenAI schema export
# ---------------------------------------------------------------------------
class TestOpenAISchemaExport:
    def test_openai_tools_schema_shape(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        schema = reg.openai_tools_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "add"
        assert schema[0]["function"]["description"] == "Add two integers."
        assert (
            schema[0]["function"]["parameters"]["properties"]["x"]["type"]
            == "integer"
        )

    def test_empty_registry_returns_empty_list(self):
        reg = ToolRegistry()
        assert reg.openai_tools_schema() == []

    def test_descriptors_contain_name_description_parameters(self):
        reg = ToolRegistry()
        reg.register_tool(
            add,
            name="add",
            description="Add two integers.",
            schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        descs = reg.descriptors()
        assert len(descs) == 1
        assert set(descs[0].keys()) == {"name", "description", "parameters"}


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------
class TestToolDataclass:
    def test_to_openai_schema(self):
        tool = Tool(
            name="foo",
            description="Does foo.",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            func=lambda x: x,
        )
        s = tool.to_openai_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "foo"
        assert s["function"]["description"] == "Does foo."

    def test_descriptor(self):
        tool = Tool(
            name="foo",
            description="Does foo.",
            parameters={"type": "object"},
            func=lambda: None,
        )
        d = tool.descriptor()
        assert d == {"name": "foo", "description": "Does foo.", "parameters": {"type": "object"}}

    def test_validate_kwargs_passes_when_no_schema(self):
        tool = Tool(name="foo", description="...", parameters={}, func=lambda: None)
        tool.validate_kwargs({"anything": True})  # should not raise
