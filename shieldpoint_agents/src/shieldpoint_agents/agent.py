"""
Agent base class — Think/Plan/Act (ReAct) loop with structured output parsing,
confidence scoring, and HITL escalation (SHLD-14).

Design
------

The :class:`Agent` runs a classic ReAct loop:

1. **Think** — Prompt the LLM with the claim, the conversation so far, and
   the list of registered tools. Ask for a JSON response conforming to
   :class:`ReActStep` (thought / action / action_input).
2. **Act** — If ``action == "FINAL_ANSWER"``, parse ``action_input`` into a
   :class:`ClaimDecision` and return. Otherwise, look up the tool in the
   :class:`ToolRegistry` and invoke it with ``action_input`` as kwargs.
3. **Observe** — Append the tool's output to the conversation history and
   loop back to Think.

Confidence & Escalation (SHLD-14)
----------------------------------

After the ReAct loop produces a ClaimDecision, the agent:

1. **Scores confidence** using :class:`ConfidenceScorer`, which combines
   the LLM's self-assessed confidence with output consistency, evidence
   grounding, and tool coverage signals.
2. **Evaluates escalation** using :class:`HITLEscalator`. If confidence is
   below 0.85, the claim is routed to manual review. If confidence is
   below 0.50, the :class:`FallbackEngine` takes over entirely.

Loop termination
----------------

The loop ends when any of these happens:

- The LLM returns ``action == "FINAL_ANSWER"`` with a valid
  :class:`ClaimDecision` — success (then confidence is evaluated).
- ``max_react_iterations`` (10 by default) is hit — fallback.
- The LLM call raises a timeout (>10s per AC) — fallback.
- The LLM call raises any other exception — fallback.
- The structured output fails to parse after ``parse_retries`` corrective
  retries — fallback.

In every fallback case, the :class:`FallbackEngine` is invoked and the
result envelope has ``source="fallback"`` with a ``fallback_reason``
explaining what went wrong.

Tracing
-------

Every LLM call is wrapped with ``@tracer.llm_call("react_think")`` so
Langfuse captures the prompt, completion, latency, and token count. Every
tool invocation is wrapped with ``@tracer.tool_call(name)``. The whole
agent run is wrapped in a ``tracer.trace("agent_run")`` context manager so
all spans attach to a single trace.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from pydantic import ValidationError

from .confidence import ConfidenceReport, ConfidenceScorer
from .config import AgentConfig
from .escalation import EscalationRecord, HITLEscalator
from .fallback import FallbackEngine
from .schemas import AgentRunResult, ClaimDecision, ReActStep
from .tools import ToolInvocationError, ToolNotFoundError, ToolRegistry
from .tracer import LangfuseTracer

logger = logging.getLogger("shieldpoint_agents.agent")


class AgentError(RuntimeError):
    """Raised when an unrecoverable error occurs in the agent (after fallback)."""


# ---------------------------------------------------------------------------
# System prompt — instructs the LLM to produce ReActStep JSON.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a ShieldPoint claims-processing agent. You operate in a ReAct loop:
Think, then choose an action, then observe the result.

Available tools:
{tools_block}

For each step, respond with ONLY a JSON object of this exact shape:
{{
  "thought": "<your reasoning>",
  "action": "<tool name from the list above, or 'FINAL_ANSWER'>",
  "action_input": {{ <arguments for the tool, or the final decision> }}
}}

When you have enough information to decide the claim, set
"action": "FINAL_ANSWER" and "action_input" to:
{{
  "decision": "approve" | "deny" | "route_to_manual_review",
  "reasoning": "<human-readable explanation>",
  "confidence": <float in [0, 1]>,
  "evidence": ["<fact 1>", "<fact 2>", ...]
}}

IMPORTANT:
- Always validate the policy before making a decision.
- Always check claim history for frequent claimants.
- Set confidence based on how certain you are: 0.95+ for very certain,
  0.85-0.95 for fairly certain, 0.50-0.85 for uncertain, below 0.50
  for very uncertain.
- If you are uncertain, set decision to "route_to_manual_review".

Do not include any text outside the JSON object. Do not wrap the JSON in
markdown code fences.
"""


class Agent:
    """Base agent class. Subclass to add custom tools or prompt sections.

    Parameters
    ----------
    name : str
        Human-readable agent name (used in traces and logs).
    tools : ToolRegistry
        Registry of tools the agent may invoke. Must already be populated.
    tracer : LangfuseTracer
        Per-agent Langfuse façade. If ``None``, a default is constructed.
    fallback : FallbackEngine
        Rule-based fallback engine. If ``None``, a default is constructed.
    config : AgentConfig, optional
        Runtime config. Defaults to :meth:`AgentConfig.from_env`.
    llm_client : Any, optional
        Pre-built OpenAI-compatible client. If ``None``, one is built from
        ``config`` on first use. Tests pass a mock here.
    system_prompt : str, optional
        Override the default system prompt (rarely needed).
    confidence_scorer : ConfidenceScorer, optional
        Multi-signal confidence scorer. If ``None``, a default is built.
    hitl_escalator : HITLEscalator, optional
        HITL escalation handler. If ``None``, a default is built from config.
    """

    def __init__(
        self,
        *,
        name: str,
        tools: ToolRegistry,
        tracer: Optional[LangfuseTracer] = None,
        fallback: Optional[FallbackEngine] = None,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        hitl_escalator: Optional[HITLEscalator] = None,
    ) -> None:
        self.name = name
        self.tools = tools
        self.tracer = tracer or LangfuseTracer(agent_name=name)
        self.fallback = fallback or FallbackEngine()
        self.config = config or AgentConfig.from_env()
        self._llm_client = llm_client
        self._system_prompt = system_prompt or _SYSTEM_PROMPT
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.hitl_escalator = hitl_escalator or HITLEscalator(
            hitl_threshold=self.config.hitl_confidence_threshold,
            fallback_threshold=self.config.fallback_confidence_threshold,
        )

        # Attach the tracer to the registry so tool invocations get traced.
        # Only override if the registry didn't already have one.
        if self.tools.tracer is None:
            self.tools.tracer = self.tracer

    # ------------------------------------------------------------------ #
    #  Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def run(self, claim: dict[str, Any]) -> AgentRunResult:
        """Run the ReAct loop on ``claim`` and return an :class:`AgentRunResult`.

        Wraps the entire loop in a Langfuse trace. If any failure occurs
        (LLM timeout, parse error, max iterations exceeded), the
        :class:`FallbackEngine` is invoked and the result has
        ``source="fallback"``.
        """
        claim_id = claim.get("claim_id")
        logger.info("Agent '%s' starting run for claim %s", self.name, claim_id)

        with self.tracer.trace(
            "agent_run",
            user_id=claim.get("adjuster_id"),
            session_id=claim.get("session_id"),
            metadata={"claim_id": claim_id, "agent.name": self.name},
            tags=[self.name],
        ) as span:
            trace_id = getattr(span, "id", None) if span else None
            try:
                return self._run_react_loop(claim, trace_id=trace_id)
            except _FallbackSignal as exc:
                # Explicit fallback request from inside the loop.
                logger.warning(
                    "Agent '%s' falling back for claim %s: %s",
                    self.name, claim_id, exc.reason,
                )
                return self.fallback.run(
                    claim,
                    agent_name=self.name,
                    fallback_reason=exc.reason,
                )
            except Exception as exc:
                # Unexpected error — log and fall back.
                logger.exception(
                    "Agent '%s' hit unexpected error for claim %s: %s",
                    self.name, claim_id, exc,
                )
                return self.fallback.run(
                    claim,
                    agent_name=self.name,
                    fallback_reason=f"unexpected_error: {exc!r}",
                )

    # ------------------------------------------------------------------ #
    #  ReAct loop                                                         #
    # ------------------------------------------------------------------ #
    def _run_react_loop(
        self, claim: dict[str, Any], *, trace_id: Optional[str]
    ) -> AgentRunResult:
        messages = self._build_initial_messages(claim)
        step_history: list[ReActStep] = []
        tools_invoked: list[str] = []

        for iteration in range(1, self.config.max_react_iterations + 1):
            logger.debug(
                "Agent '%s' iteration %d/%d",
                self.name, iteration, self.config.max_react_iterations,
            )

            # ---- THINK -------------------------------------------------
            step = self._think(messages, iteration=iteration)
            step_history.append(step)
            messages.append(
                {"role": "assistant", "content": step.model_dump_json()}
            )

            # ---- TERMINATION CHECK ------------------------------------
            if step.is_final:
                decision = self._parse_final_decision(step.action_input)
                logger.info(
                    "Agent '%s' completed in %d iteration(s). decision=%s "
                    "confidence=%.3f",
                    self.name, iteration, decision.decision, decision.confidence,
                )

                # ---- SHLD-14: Confidence scoring & HITL escalation ----
                result = self._evaluate_confidence_and_escalate(
                    decision=decision,
                    step_history=step_history,
                    tools_invoked=tools_invoked,
                    claim=claim,
                    iterations=iteration,
                    trace_id=trace_id,
                )
                return result

            # ---- ACT --------------------------------------------------
            try:
                tool_output = self.tools.invoke(step.action, **step.action_input)
                tools_invoked.append(step.action)
            except ToolNotFoundError as exc:
                # LLM hallucinated a tool — feed the error back and retry.
                obs = f"ERROR: {exc}"
                logger.warning(
                    "Agent '%s' referenced unknown tool '%s' on iter %d",
                    self.name, step.action, iteration,
                )
            except (ToolInvocationError, Exception) as exc:
                obs = f"ERROR: tool '{step.action}' failed: {exc!r}"
                logger.warning(
                    "Agent '%s' tool '%s' failed on iter %d: %s",
                    self.name, step.action, iteration, exc,
                )
            else:
                obs = self._stringify_tool_output(tool_output)

            # ---- OBSERVE ----------------------------------------------
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        # Max iterations exhausted — fall back.
        raise _FallbackSignal(
            f"max_react_iterations ({self.config.max_react_iterations}) exceeded"
        )

    # ------------------------------------------------------------------ #
    #  Confidence evaluation & HITL escalation (SHLD-14)                  #
    # ------------------------------------------------------------------ #
    def _evaluate_confidence_and_escalate(
        self,
        decision: ClaimDecision,
        *,
        step_history: list[ReActStep],
        tools_invoked: list[str],
        claim: dict[str, Any],
        iterations: int,
        trace_id: Optional[str],
    ) -> AgentRunResult:
        """Score confidence, evaluate HITL escalation, and build result.

        This is the core SHLD-14 logic:
        1. Score the decision using the multi-signal ConfidenceScorer.
        2. Evaluate whether escalation is needed via HITLEscalator.
        3. If confidence < fallback_threshold, invoke the FallbackEngine.
        4. If confidence < HITL threshold, route to manual review.
        5. Otherwise, accept the LLM's decision.
        """
        # Step 1: Score confidence
        confidence_report = self.confidence_scorer.score(
            decision,
            step_history=step_history,
            tools_invoked=tools_invoked,
            claim=claim,
        )

        # Log the confidence report to Langfuse
        self._log_confidence_to_tracer(confidence_report, claim)

        # Step 2: Evaluate escalation
        adjusted_decision, escalation_record = self.hitl_escalator.evaluate(
            decision,
            confidence_report,
            claim_id=claim.get("claim_id"),
        )

        # Step 3: Check if full fallback is needed
        if self.hitl_escalator.should_fallback(escalation_record):
            logger.warning(
                "Confidence %.3f below fallback threshold — invoking "
                "FallbackEngine for claim %s",
                confidence_report.final_score,
                claim.get("claim_id"),
            )
            fallback_result = self.fallback.run(
                claim,
                agent_name=self.name,
                fallback_reason=(
                    f"confidence_score {confidence_report.final_score:.3f} "
                    f"< fallback_threshold {self.config.fallback_confidence_threshold:.3f}"
                ),
            )
            # Preserve SHLD-14 metadata on the fallback result
            fallback_result.confidence_score = confidence_report.final_score
            fallback_result.tools_invoked = tools_invoked
            return fallback_result

        # Step 4: Build the result based on escalation status
        if escalation_record is not None:
            # HITL escalation — decision was adjusted to route_to_manual_review
            self._log_escalation_to_tracer(escalation_record)
            return AgentRunResult(
                agent_name=self.name,
                claim_id=claim.get("claim_id"),
                decision=adjusted_decision,
                source="hitl_escalation",
                iterations=iterations,
                fallback_reason=None,
                trace_id=trace_id,
                confidence_score=confidence_report.final_score,
                hitl_escalated=True,
                original_decision=escalation_record.original_decision,
                tools_invoked=tools_invoked,
            )

        # Step 5: LLM decision accepted with sufficient confidence
        return AgentRunResult(
            agent_name=self.name,
            claim_id=claim.get("claim_id"),
            decision=adjusted_decision,
            source="llm",
            iterations=iterations,
            fallback_reason=None,
            trace_id=trace_id,
            confidence_score=confidence_report.final_score,
            hitl_escalated=False,
            original_decision=None,
            tools_invoked=tools_invoked,
        )

    def _log_confidence_to_tracer(
        self, report: ConfidenceReport, claim: dict[str, Any]
    ) -> None:
        """Log the confidence report as Langfuse span metadata."""
        try:
            # Update the current trace with confidence metadata
            # (The LangfuseTracer's trace context manager should support
            # span.update() for adding metadata.)
            logger.info(
                "Confidence report for claim %s: score=%.3f "
                "self=%.3f consistency=%.3f evidence=%.3f tool_cov=%.3f flags=%s",
                claim.get("claim_id"),
                report.final_score,
                report.self_assessed,
                report.consistency,
                report.evidence_grounding,
                report.tool_coverage,
                report.flags,
            )
        except Exception:
            # Never let tracing failures affect agent behavior
            pass

    def _log_escalation_to_tracer(self, record: EscalationRecord) -> None:
        """Log the escalation record to Langfuse for audit."""
        try:
            logger.info(
                "HITL escalation: original=%s confidence=%.3f "
                "adjusted=%.3f reason=%s flags=%s",
                record.original_decision,
                record.original_confidence,
                record.adjusted_confidence,
                record.reason,
                record.confidence_flags,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  LLM call + structured output parsing                              #
    # ------------------------------------------------------------------ #
    def _think(self, messages: list[dict[str, Any]], *, iteration: int) -> ReActStep:
        """Call the LLM and parse the response into a :class:`ReActStep`.

        On parse failure, retries up to ``config.parse_retries`` times with
        a corrective prompt. After that, raises :class:`_FallbackSignal`.
        """
        last_error: Optional[str] = None
        for attempt in range(self.config.parse_retries + 1):
            prompt_messages = list(messages)
            if last_error:
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid JSON "
                            f"conforming to the ReActStep schema: {last_error}. "
                            "Please respond again with ONLY the JSON object."
                        ),
                    }
                )

            raw = self._call_llm(prompt_messages, iteration=iteration, attempt=attempt)
            try:
                step = self._parse_react_step(raw)
                return step
            except _ParseError as exc:
                last_error = str(exc)
                logger.warning(
                    "Agent '%s' iter %d attempt %d: parse error — %s",
                    self.name, iteration, attempt + 1, last_error,
                )

        raise _FallbackSignal(
            f"failed to parse LLM output after {self.config.parse_retries + 1} "
            f"attempt(s); last error: {last_error}"
        )

    def _call_llm(
        self, messages: list[dict[str, Any]], *, iteration: int, attempt: int
    ) -> str:
        """Invoke the LLM via the OpenAI-compatible client.

        Wrapped in the ``react_think`` Langfuse decorator so each call is
        traced. Times out at ``config.llm_timeout_sec`` (10s default per AC).
        On timeout or any other failure, raises :class:`_FallbackSignal`.
        """
        client = self._get_llm_client()

        @self.tracer.llm_call(f"react_think_iter{iteration}_att{attempt}")
        def _do_call() -> str:
            try:
                resp = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens_per_step,
                    timeout=self.config.llm_timeout_sec,
                )
            except Exception as exc:
                # The OpenAI SDK raises ``APITimeoutError`` (subclass of
                # ``APIConnectionError``) on timeout. We treat all exceptions
                # as fallback triggers per the AC.
                raise _FallbackSignal(
                    f"LLM call failed (iter={iteration}, attempt={attempt}): {exc!r}"
                ) from exc

            try:
                return resp.choices[0].message.content or ""
            except (AttributeError, IndexError, TypeError) as exc:
                raise _FallbackSignal(
                    f"LLM response shape unexpected: {exc!r}"
                ) from exc

        return _do_call()

    # ------------------------------------------------------------------ #
    #  Parsing helpers                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_react_step(raw: str) -> ReActStep:
        """Parse the LLM's raw text response into a :class:`ReActStep`.

        Strips markdown code fences if present, then validates with Pydantic.
        Raises :class:`_ParseError` on any failure (caller retries or falls
        back).
        """
        if not raw or not raw.strip():
            raise _ParseError("empty LLM response")

        cleaned = Agent._strip_code_fences(raw).strip()

        # The LLM should produce a single JSON object. If there's leading or
        # trailing prose, try to extract the first {...} block.
        if not cleaned.startswith("{"):
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                cleaned = match.group(0)
            else:
                raise _ParseError(
                    f"no JSON object found in response (first 200 chars: "
                    f"{cleaned[:200]!r})"
                )

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise _ParseError(f"invalid JSON: {exc.msg} at pos {exc.pos}") from exc

        try:
            return ReActStep.model_validate(obj)
        except ValidationError as exc:
            raise _ParseError(
                f"schema validation failed: {exc.errors()[:3]}"
            ) from exc

    @staticmethod
    def _parse_final_decision(action_input: dict[str, Any]) -> ClaimDecision:
        """Validate the FINAL_ANSWER payload as a :class:`ClaimDecision`."""
        try:
            return ClaimDecision.model_validate(action_input)
        except ValidationError as exc:
            # The LLM said FINAL_ANSWER but the payload doesn't match the
            # ClaimDecision schema — caller should fall back.
            raise _FallbackSignal(
                f"FINAL_ANSWER payload failed ClaimDecision validation: "
                f"{exc.errors()[:3]}"
            ) from exc

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` fences if present."""
        match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
        return match.group(1) if match else text

    @staticmethod
    def _stringify_tool_output(output: Any) -> str:
        """Convert a tool's return value to a string for the next prompt."""
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, default=str, indent=2)
        except (TypeError, ValueError):
            return str(output)

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                #
    # ------------------------------------------------------------------ #
    def _build_initial_messages(self, claim: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the opening messages list (system + first user turn)."""
        tools_block = self._render_tools_block()
        system = self._system_prompt.format(tools_block=tools_block)
        claim_json = json.dumps(claim, default=str, indent=2)
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Process the following claim through the ReAct loop. "
                    f"When you have enough information, return FINAL_ANSWER.\n\n"
                    f"Claim:\n{claim_json}"
                ),
            },
        ]

    def _render_tools_block(self) -> str:
        """Render the available tools as a bullet list for the system prompt."""
        if not self.tools.names():
            return "(no tools registered)"
        lines = []
        for tool in self.tools:
            params = ", ".join(
                tool.parameters.get("properties", {}).keys()
            ) or "(no params)"
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  LLM client lazy-init                                               #
    # ------------------------------------------------------------------ #
    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            from ._lmstudio import build_lm_studio_client

            self._llm_client = build_lm_studio_client(self.config)
        return self._llm_client


# ---------------------------------------------------------------------------
# Internal signals — control flow inside the agent, not raised to callers.
# ---------------------------------------------------------------------------
class _FallbackSignal(Exception):
    """Raised internally to break out of the ReAct loop and trigger fallback."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _ParseError(Exception):
    """Raised when the LLM's response can't be parsed into a ReActStep."""
