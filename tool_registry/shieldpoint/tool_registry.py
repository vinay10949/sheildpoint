"""
ToolRegistry — the standardized interface between the ShieldPoint agent and
external systems.

Each tool is a Python function paired with a JSON-Schema descriptor that the
LLM uses to decide which tool to invoke and with what parameters. On
``invoke()`` the registry:

1. Looks up the tool by name (raises :class:`ToolNotFoundError` if absent).
2. Validates the kwargs against the tool's JSON Schema using
   ``jsonschema.Draft7Validator`` (raises :class:`ToolValidationError` with a
   structured error message on mismatch).
3. Opens a Langfuse span via the injected :class:`SpanRecorder` capturing
   tool name, input kwargs, output, latency, and any exception.
4. Invokes the underlying function.
5. Returns the result, or re-raises a :class:`ToolInvocationError` wrapping
   the original exception (after the span has been recorded).

The registry also exposes ``openai_tools_schema()`` — a list of tool
descriptors in OpenAI function-calling format — so the agent can present
available tools to the LLM when prompting for the next ReAct step.

Design notes
------------
- ``register_tool`` / ``get_tool`` / ``invoke`` are the public method names
  required by the SP-201 acceptance criteria. The legacy ``register`` / ``get``
  aliases are kept for backwards compatibility with the existing repo.
- The ``SpanRecorder`` is constructor-injected so tests can pass
  :class:`NullSpanRecorder` and assert on captured spans without needing a
  live Langfuse server.
- Schema validation runs **before** the function is called, so a buggy tool
  never sees invalid arguments. Validation errors are returned as structured
  dicts (not bare strings) so the agent can feed them back to the LLM.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import jsonschema

from .langfuse_span import NullSpanRecorder, SpanRecorder

logger = logging.getLogger("shieldpoint.tool_registry")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ToolNotFoundError(KeyError):
    """Raised when ``invoke()`` is called on a tool that isn't registered."""


class ToolValidationError(ValueError):
    """Raised when kwargs fail JSON-Schema validation before invocation.

    Carries a structured ``details`` dict so callers (the agent loop) can
    feed the error back to the LLM as a corrective prompt.
    """

    def __init__(self, message: str, *, tool: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.tool = tool
        self.details = details


class ToolInvocationError(RuntimeError):
    """Wraps any exception raised by the underlying tool function.

    The original exception is preserved on ``__cause__`` so the agent can
    introspect it. ``tool`` and ``input`` are exposed for structured logging.
    """

    def __init__(
        self,
        message: str,
        *,
        tool: str,
        input: Optional[dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.tool = tool
        self.input = input or {}
        if cause is not None:
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# Tool dataclass
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """A registered tool — function + JSON-Schema descriptor + metadata."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    # ------------------------------------------------------------------ #
    #  Validation                                                        #
    # ------------------------------------------------------------------ #
    def validate_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate ``kwargs`` against ``self.parameters`` (Draft 7).

        Raises :class:`jsonschema.ValidationError` on mismatch. The caller
        (ToolRegistry) is responsible for translating this into a
        :class:`ToolValidationError` with structured details.
        """
        if not self.parameters:
            return
        jsonschema.Draft7Validator(self.parameters).validate(kwargs)

    # ------------------------------------------------------------------ #
    #  Invocation                                                        #
    # ------------------------------------------------------------------ #
    def invoke(self, **kwargs: Any) -> Any:
        """Validate args against ``self.parameters`` and call ``self.func``."""
        self.validate_kwargs(kwargs)
        return self.func(**kwargs)

    # ------------------------------------------------------------------ #
    #  Schema export                                                     #
    # ------------------------------------------------------------------ #
    def to_openai_schema(self) -> dict[str, Any]:
        """Return this tool in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def descriptor(self) -> dict[str, Any]:
        """Return a serializable descriptor (name + description + schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------
@dataclass
class ToolRegistry:
    """Registry of tools available to a ShieldPoint agent.

    Parameters
    ----------
    span_recorder:
        Object responsible for opening Langfuse spans around tool calls.
        Defaults to :class:`NullSpanRecorder` (no-op) so the registry works
        out-of-the-box in tests / when Langfuse is unreachable.

    Usage
    -----
    ::

        registry = ToolRegistry()

        @registry.register_tool(
            name="policy_validate",
            description="Check policy status and coverage limits.",
            schema={
                "type": "object",
                "properties": {"policy_id": {"type": "string"}},
                "required": ["policy_id"],
                "additionalProperties": False,
            },
        )
        def policy_validate(policy_id: str) -> dict:
            ...

        result = registry.invoke("policy_validate", policy_id="HO-2024-001")
    """

    span_recorder: SpanRecorder = field(default_factory=NullSpanRecorder)
    _tools: dict[str, Tool] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Registration                                                       #
    # ------------------------------------------------------------------ #
    def register_tool(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[dict[str, Any]] = None,
    ) -> Callable[..., Any] | Tool:
        """Register a function as a tool.

        Can be used as a decorator (with kwargs) or called directly::

            @registry.register_tool(name="foo", description="...", schema={...})
            def foo(...): ...

            # or
            registry.register_tool(my_func, name="bar", description="...", schema={...})

        Returns the original function (decorator use) or the registered
        :class:`Tool` (direct-call use).
        """
        return self._register(func, name=name, description=description, schema=schema)

    # Backwards-compatible alias for the existing repo's API.
    register = register_tool

    def _register(
        self,
        func: Optional[Callable[..., Any]],
        *,
        name: Optional[str],
        description: Optional[str],
        schema: Optional[dict[str, Any]],
    ) -> Callable[..., Any] | Tool:
        def _do_register(fn: Callable[..., Any]) -> Tool:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip().split("\n", 1)[0]
            if not tool_desc:
                raise ValueError(
                    f"Tool '{tool_name}' has no description. Provide one via "
                    "the `description` kwarg or a function docstring."
                )
            inferred = schema if schema is not None else self._infer_schema(fn)
            tool = Tool(
                name=tool_name,
                description=tool_desc,
                parameters=inferred,
                func=fn,
            )
            self._tools[tool_name] = tool
            logger.debug("Registered tool '%s' (%s)", tool_name, tool_desc)
            return tool

        if func is None:
            # Decorator-with-args usage: return a decorator.
            def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
                _do_register(fn)
                return fn

            return decorator

        return _do_register(func)

    @staticmethod
    def _infer_schema(fn: Callable[..., Any]) -> dict[str, Any]:
        """Infer a permissive JSON Schema from the function's signature.

        Handles both runtime type objects (``int``, ``str``) and string
        annotations (which are what ``from __future__ import annotations``
        produces under PEP 563).
        """
        # Map both the type object AND its string name to the JSON type.
        py_to_json: dict[Any, str] = {
            str: "string", "str": "string",
            int: "integer", "int": "integer",
            float: "number", "float": "number",
            bool: "boolean", "bool": "boolean",
            dict: "object", "dict": "object",
            list: "array", "list": "array",
        }
        properties: dict[str, Any] = {}
        required: list[str] = []
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = param.annotation
            # Default to "string" for un-annotated or unknown annotations.
            json_type = py_to_json.get(annotation, "string")
            properties[pname] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    # ------------------------------------------------------------------ #
    #  Lookup                                                            #
    # ------------------------------------------------------------------ #
    def get_tool(self, name: str) -> Tool:
        """Return the :class:`Tool` registered under ``name``.

        Raises :class:`ToolNotFoundError` if no such tool exists.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not registered. Available: {list(self._tools)}"
            )
        return self._tools[name]

    # Backwards-compatible alias.
    get = get_tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    # ------------------------------------------------------------------ #
    #  OpenAI-format schema export                                       #
    # ------------------------------------------------------------------ #
    def openai_tools_schema(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def descriptors(self) -> list[dict[str, Any]]:
        """Return serializable descriptors for every registered tool."""
        return [t.descriptor() for t in self._tools.values()]

    # ------------------------------------------------------------------ #
    #  Invocation                                                        #
    # ------------------------------------------------------------------ #
    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        """Validate args, open a Langfuse span, and invoke the named tool.

        Errors
        ------
        - :class:`ToolNotFoundError`     — no tool with that name.
        - :class:`ToolValidationError`   — kwargs failed JSON-Schema validation.
        - :class:`ToolInvocationError`   — the underlying function raised.

        All three are recorded on the Langfuse span (input / output / error)
        before being re-raised.
        """
        try:
            tool = self.get_tool(name)
        except ToolNotFoundError:
            raise

        # ---- Schema validation (before opening the span) -------------- #
        try:
            tool.validate_kwargs(kwargs)
        except jsonschema.ValidationError as exc:
            error = ToolValidationError(
                f"Tool '{name}' argument validation failed: {exc.message}",
                tool=name,
                details={
                    "tool": name,
                    "input": kwargs,
                    "json_path": exc.json_path,
                    "schema_path": list(exc.schema_path),
                    "message": exc.message,
                    "validator": exc.validator,
                    "validator_value": exc.validator_value,
                },
            )
            # Still record the failed attempt so the audit trail shows it.
            self.span_recorder.record_failed_tool_call(
                name=name,
                input=kwargs,
                error=error,
            )
            logger.warning("Tool '%s' arg validation failed: %s", name, exc.message)
            raise error

        # ---- Invocation inside a Langfuse span ------------------------ #
        started_at = time.perf_counter()
        span_handle = self.span_recorder.start_tool_span(
            name=name,
            input=kwargs,
        )
        try:
            result = tool.func(**kwargs)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            wrapped = ToolInvocationError(
                f"Tool '{name}' failed: {exc!r}",
                tool=name,
                input=kwargs,
                cause=exc,
            )
            self.span_recorder.end_tool_span(
                handle=span_handle,
                name=name,
                input=kwargs,
                output=None,
                error=wrapped,
                latency_ms=latency_ms,
            )
            logger.exception("Tool '%s' raised an exception", name)
            raise wrapped from exc
        else:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            self.span_recorder.end_tool_span(
                handle=span_handle,
                name=name,
                input=kwargs,
                output=result,
                error=None,
                latency_ms=latency_ms,
            )
            return result
