"""
ToolRegistry — registers Python callables with JSON-Schema descriptors.

Each tool is a regular Python function paired with a JSON Schema describing
its parameters. On invocation, the registry:

1. Validates the kwargs against the schema (using ``jsonschema``).
2. Logs the call (name, input, output, latency, error) to Langfuse via the
   ``LangfuseTracer.tool_call`` decorator.
3. Invokes the underlying function.
4. Returns the result (or re-raises the exception after logging).

The registry also exposes ``openai_tools_schema()`` — a list of tool
descriptors in the OpenAI function-calling format — so the Agent can present
available tools to the LLM when prompting for the next ReAct step.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import jsonschema

from .tracer import LangfuseTracer

logger = logging.getLogger("shieldpoint_agents.tools")


@dataclass
class Tool:
    """A registered tool — function + JSON-Schema descriptor."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def invoke(self, **kwargs: Any) -> Any:
        """Validate args against ``self.parameters`` and call ``self.func``."""
        if self.parameters:
            # ``jsonschema.validate`` raises ``ValidationError`` on mismatch.
            # We use the ``Draft7Validator`` for explicit version pinning.
            jsonschema.Draft7Validator(self.parameters).validate(kwargs)
        return self.func(**kwargs)

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


class ToolNotFoundError(KeyError):
    """Raised when the agent tries to invoke a tool that isn't registered."""


class ToolInvocationError(RuntimeError):
    """Raised when a registered tool raises an exception during invocation."""


@dataclass
class ToolRegistry:
    """Registry of tools available to an :class:`Agent`.

    Usage::

        registry = ToolRegistry(tracer=LangfuseTracer(agent_name="claim-agent"))

        @registry.register(
            name="validate_policy",
            description="Look up a policy and return coverage limits.",
            schema={
                "type": "object",
                "properties": {
                    "policy_id": {"type": "string"},
                },
                "required": ["policy_id"],
            },
        )
        def validate_policy(policy_id: str) -> dict:
            ...

    Or, equivalently, call :meth:`register` directly with a function and
    schema dict.
    """

    tracer: Optional[LangfuseTracer] = None
    _tools: dict[str, Tool] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #  Registration                                                       #
    # ------------------------------------------------------------------ #
    def register(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[dict[str, Any]] = None,
    ) -> Callable[..., Any] | Tool:
        """Register a function as a tool.

        Can be used as a decorator (with kwargs) or called directly::

            @registry.register(name="foo", description="...", schema={...})
            def foo(...): ...

            # or
            registry.register(my_func, name="bar", description="...", schema={...})

        Returns the original function (decorator use) or the registered
        :class:`Tool` (direct-call use).
        """
        def _do_register(fn: Callable[..., Any]) -> Tool:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip().split("\n", 1)[0]
            if not tool_desc:
                raise ValueError(
                    f"Tool '{tool_name}' has no description. Provide one via "
                    "the `description` kwarg or a function docstring."
                )
            if schema is None:
                # Infer a permissive schema if none was provided.
                inferred = self._infer_schema(fn)
            else:
                inferred = schema

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

        # Direct call usage.
        return _do_register(func)

    @staticmethod
    def _infer_schema(fn: Callable[..., Any]) -> dict[str, Any]:
        """Infer a JSON Schema from the function's signature.

        Only handles the common cases — strings, ints, floats, bools, and
        ``dict[str, Any]`` / ``list`` parameters. For richer schemas, pass
        an explicit ``schema=`` to :meth:`register`.
        """
        py_to_json: dict[str, str] = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
        }
        properties: dict[str, Any] = {}
        required: list[str] = []
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = param.annotation
            json_type = "string"  # default
            if annotation in (str, "str"):
                json_type = "string"
            elif annotation in (int, "int"):
                json_type = "integer"
            elif annotation in (float, "float"):
                json_type = "number"
            elif annotation in (bool, "bool"):
                json_type = "boolean"
            elif annotation in (dict, "dict"):
                json_type = "object"
            elif annotation in (list, "list"):
                json_type = "array"
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
    #  Lookup                                                             #
    # ------------------------------------------------------------------ #
    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFoundError(
                f"Tool '{name}' not registered. Available: {list(self._tools)}"
            )
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)

    # ------------------------------------------------------------------ #
    #  OpenAI-format schema export                                        #
    # ------------------------------------------------------------------ #
    def openai_tools_schema(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    # ------------------------------------------------------------------ #
    #  Invocation                                                         #
    # ------------------------------------------------------------------ #
    def invoke(self, name: str, /, **kwargs: Any) -> Any:
        """Validate args, decorate with tracer, and invoke the named tool.

        Errors from the underlying function are wrapped in
        :class:`ToolInvocationError` after being logged via the tracer.
        """
        tool = self.get(name)

        # Wrap the actual invocation in the tool_call decorator if a tracer
        # is attached. The decorator captures name, args, result, error.
        if self.tracer is not None:
            decorated = self.tracer.tool_call(name)(tool.invoke)
        else:
            decorated = tool.invoke

        try:
            return decorated(**kwargs)
        except jsonschema.ValidationError as exc:
            # Schema validation failure — caller (Agent) usually retries
            # the LLM step with a corrective prompt.
            logger.warning(
                "Tool '%s' arg validation failed: %s", name, exc.message
            )
            raise
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            raise ToolInvocationError(
                f"Tool '{name}' failed: {exc!r}"
            ) from exc
