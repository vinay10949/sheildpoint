"""
ShieldPoint Observability Subpackage
====================================

Thin wrapper around the Langfuse Python SDK (``langfuse>=2.0``) that
decorates every LLM call, tool invocation, and span to capture:

- input prompts (messages, model, params)
- model responses (content, finish_reason)
- latency (wall-clock ms)
- token counts (prompt, completion, total — extracted from the OpenAI-style
  ``response.usage`` field returned by LM Studio)
- tool invocations (function name, args, result, error)
- errors and exceptions (with stack trace metadata)

All traces are sent to the self-hosted Langfuse instance running inside the
ShieldPoint network (``LANGFUSE_HOST``). No data ever leaves the network
boundary — the Langfuse SDK uses an HTTP POST to ``${LANGFUSE_HOST}`` which
resolves to ``langfuse:3000`` on the ``shieldpoint-net`` Docker bridge.

Public API
----------
    from shieldpoint.observability import observe_llm, observe_tool, tracer

    @observe_llm(name="claim_classify")
    def classify_claim(claim_text: str) -> dict:
        client = OpenAI(base_url=..., api_key=...)
        return client.chat.completions.create(model=..., messages=[...])

    @observe_tool(name="policy_validate")
    def policy_validate(policy_id: str, claim: dict) -> dict:
        ...
"""

from .langfuse_wrapper import (
    ShieldPointTracer,
    tracer,                # module-level singleton instance
    observe_llm,           # decorator for LLM calls
    observe_tool,          # decorator for tool calls
    trace_context,         # context manager for explicit trace boundaries
    current_trace_id,      # accessor for the active trace ID (if any)
    LangfuseNotConfiguredError,
)

__all__ = [
    "ShieldPointTracer",
    "tracer",
    "observe_llm",
    "observe_tool",
    "trace_context",
    "current_trace_id",
    "LangfuseNotConfiguredError",
]
