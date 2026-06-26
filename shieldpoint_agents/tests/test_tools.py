"""Unit tests for ToolRegistry — schema validation, invocation, OpenAI export."""

from __future__ import annotations

import pytest

from shieldpoint_agents import ToolRegistry
from shieldpoint_agents.tools import ToolNotFoundError, ToolInvocationError


# ---------------------------------------------------------------------------
# Test fixtures: simple tools used across the file.
# ---------------------------------------------------------------------------
def add(x: int, y: int = 0) -> int:
    """Add two integers."""
    return x + y


def boom(**kwargs):
    """A tool that always raises."""
    raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_register_with_explicit_schema(self):
        reg = ToolRegistry()
        reg.register(
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

        @reg.register(
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
        reg.register(add, name="add", description="Add two integers.")
        tool = reg.get("add")
        assert tool.parameters["properties"]["x"]["type"] == "integer"
        assert tool.parameters["properties"]["y"]["type"] == "integer"
        assert "x" in tool.parameters["required"]
        assert "y" not in tool.parameters.get("required", [])

    def test_register_requires_description(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="no description"):

            def no_doc(x):
                return x

            reg.register(no_doc, name="no_doc")


# ---------------------------------------------------------------------------
# Invocation + schema validation
# ---------------------------------------------------------------------------
class TestInvocation:
    def test_invoke_valid_args(self):
        reg = ToolRegistry()
        reg.register(
            add,
            name="add",
            description="Add two integers.",
            schema={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        )
        assert reg.invoke("add", x=3, y=4) == 7

    def test_invoke_rejects_wrong_type(self):
        reg = ToolRegistry()
        reg.register(
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
        with pytest.raises(Exception):  # jsonschema.ValidationError
            reg.invoke("add", x="not-an-int")

    def test_invoke_unknown_tool(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.invoke("nope")

    def test_invoke_wraps_function_exception(self):
        reg = ToolRegistry()
        reg.register(
            boom,
            name="boom",
            description="Always raises.",
            schema={"type": "object"},
        )
        with pytest.raises(ToolInvocationError):
            reg.invoke("boom")


# ---------------------------------------------------------------------------
# OpenAI-format export
# ---------------------------------------------------------------------------
class TestOpenAISchemaExport:
    def test_openai_tools_schema_shape(self):
        reg = ToolRegistry()
        reg.register(
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
        assert schema[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"

    def test_empty_registry_returns_empty_list(self):
        reg = ToolRegistry()
        assert reg.openai_tools_schema() == []
