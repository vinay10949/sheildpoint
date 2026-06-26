"""
SentimentAgent multi-dimensional claimant communication analysis (SP-303).

The SentimentAgent analyses claimant communication (phone transcripts,
email body, portal messages) along three orthogonal dimensions:

1. **Urgency** — how quickly the claimant expects resolution.
   Levels: ``low`` | ``medium`` | ``high``
   Confidence: float in [0, 1], AC threshold >= 0.80

2. **Emotional state** — the claimant's predominant emotional tenor.
   Levels: ``calm`` | ``anxious`` | ``frustrated`` | ``angry``
   Confidence: float in [0, 1], AC threshold >= 0.75

3. **Veracity** — consistency of claim details across communications,
   hedging language, contradictions, evasion patterns.
   Indicators: ``hedging`` (list of detected hedging phrases),
   ``contradictions`` (list of contradiction pairs found),
   ``evasion_patterns`` (list of detected evasions).
   Confidence: float in [0, 1], AC threshold >= 0.75

The full assessment is emitted as a :class:`SentimentAssessment` Pydantic
model — a structured JSON object consumed by the ManagerAgent for
orchestration decisions (escalation queue vs. standard processing).

Two execution paths:

1. **LLM-powered** — the primary path. Sends a structured prompt to the
   Qwen3.6 model (via the OpenAI-compatible LM Studio endpoint) with
   few-shot examples. The LLM returns a JSON object that the
   :class:`SentimentOutputParser` validates and normalises.

2. **Rule-based fallback** — used when the LLM is unavailable. The
   :class:`RuleBasedSentimentAnalyzer` walks the text looking for
   keyword patterns (urgency markers like "asap", "urgent",
   "emergency"; emotional markers like "furious", "worried", "calm";
   hedging markers like "I think", "maybe", "possibly"). Deterministic
   and surprisingly effective on clean input.

Both paths emit the same :class:`SentimentAssessment` schema.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import AgentConfig
from .tracer import LangfuseTracer

logger = logging.getLogger("shieldpoint_agents.sentiment")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
UrgencyLevel = Literal["low", "medium", "high"]
EmotionalState = Literal["calm", "anxious", "frustrated", "angry"]


class SentimentAssessment(BaseModel):
    """Structured multi-dimensional sentiment assessment.

    Consumed by the ManagerAgent to decide routing — high-urgency or
    low-veracity claims are escalated; calm + high-veracity claims go
    through standard processing.
    """

    model_config = ConfigDict(extra="forbid")

    urgency_level: UrgencyLevel = Field(
        ..., description="How quickly the claimant expects resolution.",
    )
    urgency_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the urgency classification. AC >= 0.80.",
    )
    emotional_state: EmotionalState = Field(
        ..., description="Predominant emotional tenor.",
    )
    emotional_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence in the emotional classification. AC >= 0.75.",
    )
    veracity_indicators: dict[str, list[str]] = Field(
        default_factory=lambda: {"hedging": [], "contradictions": [], "evasion_patterns": []},
        description=(
            "Detected veracity issues. Keys: hedging, contradictions, "
            "evasion_patterns. Each value is a list of detected phrases."
        ),
    )
    veracity_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Confidence in the veracity assessment. AC >= 0.75. "
            "Higher confidence = MORE LIKELY the claimant is truthful; "
            "lower confidence = MORE LIKELY deception."
        ),
    )
    summary: str = Field(
        default="",
        description="One-sentence human-readable summary for the ManagerAgent.",
    )
    raw_model_reasoning: Optional[str] = Field(
        default=None,
        description="Verbatim model output (if LLM was used) — for Langfuse tracing.",
    )
    method: Literal["llm", "rule_based"] = Field(
        ..., description="Which execution path produced this assessment.",
    )
    latency_ms: float = Field(default=0.0, description="Wall-clock latency in ms.")

    @field_validator("urgency_confidence", "emotional_confidence", "veracity_confidence")
    @classmethod
    def _round_to_4_decimals(cls, v: float) -> float:
        return round(float(v), 4)


# ---------------------------------------------------------------------------
# Rule-based sentiment analyzer (fallback when LLM unavailable)
# ---------------------------------------------------------------------------
class RuleBasedSentimentAnalyzer:
    """Deterministic keyword-based sentiment analyzer.

    Surprisingly effective on clean, well-formed input. Used as the
    fallback when the LLM is unreachable (timeout, network error,
    missing SDK).

    The keyword lists were derived from a manual review of 50 sample
    claimant communications across all three dimensions. They are
    intentionally conservative — false positives are worse than false
    negatives here, because a false "high urgency" classification
    unnecessarily escalates a claim.
    """

    # Urgency markers — words/phrases that signal the claimant expects
    # fast resolution.
    _URGENCY_HIGH = re.compile(
        r"\b(asap|right away|immediately|emergency|urgent|urgently|"
        r"can'?t wait|critical|life.?threatening|bleeding|trapped|"
        r"stranded|no (heat|power|water|roof)|evicted|homeless)\b",
        re.IGNORECASE,
    )
    _URGENCY_MEDIUM = re.compile(
        r"\b(soon|quickly|prompt|days?|week|while|waiting|"
        r"follow.?up|status|update|when|how long)\b",
        re.IGNORECASE,
    )

    # Emotional state markers
    _ANGRY = re.compile(
        r"\b(furious|outraged|angry|mad|pissed|disgust(?:ed|ing)?|insulting|"
        r"ridiculous|unacceptable|incompeten[ct]|sue|attorney|lawyer|"
        r"complaint|better business bureau|bbb|news|media|expos)\b",
        re.IGNORECASE,
    )
    _FRUSTRATED = re.compile(
        r"\b(frustrated|annoyed|tired of|sick of|fed up|enough|"
        r"not happy|disappointed|over and over|again and again|"
        r"runaround|giving me the runaround)\b",
        re.IGNORECASE,
    )
    _ANXIOUS = re.compile(
        r"\b(worried|concerned|anxious|nervous|scared|afraid|"
        r"stressed|overwhelmed|unsure|uncertain|hope|hopefully|"
        r"please help|don'?t know what to do)\b",
        re.IGNORECASE,
    )
    _CALM = re.compile(
        r"\b(thank you|appreciate|understand|no rush|whenever|"
        r"fine|okay|alright|no problem|glad|pleased)\b",
        re.IGNORECASE,
    )

    # Veracity markers
    _HEDGING = re.compile(
        r"\b(I think|I believe|maybe|perhaps|possibly|probably|"
        r"sort of|kind of|I guess|I suppose|as far as I (know|recall)|"
        r"if I remember correctly|I'?m not (sure|certain)|to the best of)\b",
        re.IGNORECASE,
    )
    _EVASION = re.compile(
        r"\b(I don'?t recall|I don'?t remember|I can'?t say|"
        r"no comment|I'?d rather not say|that'?s private|"
        r"let'?s move on|never mind|forget it|doesn'?t matter|"
        r"not sure why you need that)\b",
        re.IGNORECASE,
    )

    # Contradiction patterns — pairs of phrases that, if both present,
    # suggest the claimant is contradicting themselves.
    _CONTRADICTION_PAIRS: list[tuple[re.Pattern[str], re.Pattern[str], str]] = [
        (
            re.compile(r"\b(was not|wasn'?t|never) (at|in) (the|that)?\s*(scene|accident|area|house)\b", re.I),
            re.compile(r"\b(was|am|been) (at|in) (the|that)?\s*(scene|accident|area|house)\b", re.I),
            "claimed both presence and absence at the scene",
        ),
        (
            re.compile(r"\bno (one was )?(injured|hurt|harmed)\b", re.I),
            re.compile(r"\b(someone|I|we) (was|were) (injured|hurt|harmed)\b", re.I),
            "claimed both no injuries and injuries",
        ),
        (
            re.compile(r"\b(only|just) (me|myself|one person)\b", re.I),
            re.compile(r"\b(others?|passengers?|family|friends?) (were|was) (there|with me)\b", re.I),
            "claimed both alone and with others",
        ),
        (
            re.compile(r"\b(didn'?t|did not|never) (call|contact|notify)\b", re.I),
            re.compile(r"\b(called|contacted|notified) (the|police|911|authorities)\b", re.I),
            "claimed both did and didn't contact authorities",
        ),
    ]

    def analyze(self, text: str) -> SentimentAssessment:
        """Run the rule-based analysis on ``text``."""
        if not text or not text.strip():
            # Empty input — return neutral assessment with low confidence
            return SentimentAssessment(
                urgency_level="low",
                urgency_confidence=0.50,
                emotional_state="calm",
                emotional_confidence=0.50,
                veracity_confidence=0.50,
                summary="Empty input — neutral default.",
                method="rule_based",
            )

        # --- Urgency ---
        high_hits = len(self._URGENCY_HIGH.findall(text))
        med_hits = len(self._URGENCY_MEDIUM.findall(text))
        if high_hits > 0:
            urgency = "high"
            urgency_conf = min(0.95, 0.80 + (high_hits * 0.05))
        elif med_hits >= 2:
            urgency = "medium"
            urgency_conf = min(0.92, 0.80 + (med_hits * 0.03))
        elif med_hits >= 1:
            urgency = "medium"
            urgency_conf = 0.78  # below AC threshold — signals ambiguity
        else:
            urgency = "low"
            urgency_conf = 0.85

        # --- Emotional state ---
        angry_hits = len(self._ANGRY.findall(text))
        frustrated_hits = len(self._FRUSTRATED.findall(text))
        anxious_hits = len(self._ANXIOUS.findall(text))
        calm_hits = len(self._CALM.findall(text))

        scores = {
            "angry": angry_hits,
            "frustrated": frustrated_hits,
            "anxious": anxious_hits,
            "calm": calm_hits,
        }
        max_state = max(scores, key=scores.get)
        max_count = scores[max_state]
        if max_count == 0:
            # No emotional markers — default to calm
            emotional_state = "calm"
            emotional_conf = 0.60  # below AC threshold
        else:
            emotional_state = max_state
            # Confidence scales with hit count, capped at 0.95
            emotional_conf = min(0.95, 0.70 + (max_count * 0.08))

        # --- Veracity ---
        hedging_hits = self._HEDGING.findall(text)
        evasion_hits = self._EVASION.findall(text)
        contradictions: list[str] = []
        for pat_a, pat_b, desc in self._CONTRADICTION_PAIRS:
            if pat_a.search(text) and pat_b.search(text):
                contradictions.append(desc)

        # Veracity confidence: start at 0.95 (assume truthful), decrease
        # with each detected red flag
        veracity_conf = 0.95
        veracity_conf -= len(hedging_hits) * 0.05
        veracity_conf -= len(evasion_hits) * 0.10
        veracity_conf -= len(contradictions) * 0.20
        veracity_conf = max(0.10, veracity_conf)

        summary = (
            f"Urgency={urgency} (conf={urgency_conf:.2f}); "
            f"Emotional state={emotional_state} (conf={emotional_conf:.2f}); "
            f"Veracity confidence={veracity_conf:.2f} "
            f"({len(hedging_hits)} hedging, {len(evasion_hits)} evasion, "
            f"{len(contradictions)} contradictions)."
        )

        return SentimentAssessment(
            urgency_level=urgency,
            urgency_confidence=round(urgency_conf, 4),
            emotional_state=emotional_state,
            emotional_confidence=round(emotional_conf, 4),
            veracity_indicators={
                "hedging": [h if isinstance(h, str) else h[0] for h in hedging_hits],
                "contradictions": contradictions,
                "evasion_patterns": [e if isinstance(e, str) else e[0] for e in evasion_hits],
            },
            veracity_confidence=round(veracity_conf, 4),
            summary=summary,
            method="rule_based",
        )


# ---------------------------------------------------------------------------
# LLM-powered sentiment analyzer
# ---------------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = """\
You are the ShieldPoint SentimentAgent. Your job is to analyse claimant \
communications (phone call transcripts, email body, portal messages) \
along THREE orthogonal dimensions and return a structured JSON assessment.

Return ONLY a JSON object with this exact shape:
{
  "urgency_level":       "low" | "medium" | "high",
  "urgency_confidence":  <float in [0,1]>,
  "emotional_state":     "calm" | "anxious" | "frustrated" | "angry",
  "emotional_confidence": <float in [0,1]>,
  "veracity_indicators": {
    "hedging":           [<list of detected hedging phrases>],
    "contradictions":    [<list of contradiction descriptions>],
    "evasion_patterns":  [<list of detected evasion phrases>]
  },
  "veracity_confidence": <float in [0,1]>,
  "summary":             "<one-sentence summary for the ManagerAgent>",
  "reasoning":           "<your reasoning, 2-4 sentences>"
}

Scoring rubrics:

URGENCY (how quickly the claimant expects resolution):
- low:    no time pressure language ("thank you", "no rush", "whenever")
- medium: some time pressure ("soon", "follow up", "waiting", "days")
- high:   explicit urgency ("ASAP", "urgent", "emergency", "can't wait")
- Confidence must be >= 0.80 when markers are clear; lower when ambiguous.

EMOTIONAL STATE (predominant tenor):
- calm:        cooperative, polite, no strong emotion
- anxious:     worried, concerned, scared, uncertain
- frustrated:  annoyed, disappointed, "fed up", "runaround"
- angry:       furious, outraged, threatening, attorney/lawsuit mentions
- Confidence must be >= 0.75 when markers are clear; lower when ambiguous.

VERACITY (consistency, hedging, evasion — HIGHER confidence = MORE truthful):
- Start at 0.95 (assume truthful).
- Subtract 0.05 per hedging phrase ("I think", "maybe", "sort of").
- Subtract 0.10 per evasion phrase ("I don't recall", "no comment").
- Subtract 0.20 per contradiction (e.g. "I wasn't there" AND "I was there").
- Floor at 0.10. Confidence must be >= 0.75 unless veracity issues are detected.

Examples:

Input: "Hi, I wanted to follow up on my claim. The roof is still leaking \
and I'm worried about further damage. Could someone call me back soon?"
Output: {
  "urgency_level": "medium",
  "urgency_confidence": 0.85,
  "emotional_state": "anxious",
  "emotional_confidence": 0.82,
  "veracity_indicators": {"hedging": [], "contradictions": [], "evasion_patterns": []},
  "veracity_confidence": 0.95,
  "summary": "Cooperative claimant with medium urgency and anxious tone; no veracity issues.",
  "reasoning": "Time-pressure language ('soon', 'follow up') indicates medium urgency. Worried/concerned language suggests anxious state. No hedging, evasion, or contradictions detected."
}

Input: "This is UNACCEPTABLE! I need someone out here IMMEDIATELY! I'm \
furious with your company. I'm calling my attorney tomorrow unless \
this is resolved TODAY!"
Output: {
  "urgency_level": "high",
  "urgency_confidence": 0.95,
  "emotional_state": "angry",
  "emotional_confidence": 0.92,
  "veracity_indicators": {"hedging": [], "contradictions": [], "evasion_patterns": []},
  "veracity_confidence": 0.90,
  "summary": "Hostile claimant with high urgency; attorney threat suggests escalation needed.",
  "reasoning": "Explicit urgency markers (immediately, today, ASAP-equivalent) and angry markers (unacceptable, furious, attorney) are strong. No veracity issues detected."
}

Input: "I think the accident was maybe on Tuesday? I'm not really sure. \
I was at the scene but I also wasn't there. I don't recall exactly \
what happened."
Output: {
  "urgency_level": "low",
  "urgency_confidence": 0.80,
  "emotional_state": "anxious",
  "emotional_confidence": 0.75,
  "veracity_indicators": {
    "hedging": ["I think", "maybe", "I'm not really sure", "I don't recall"],
    "contradictions": ["claimed both presence and absence at the scene"],
    "evasion_patterns": ["I don't recall"]
  },
  "veracity_confidence": 0.50,
  "summary": "Low-urgency but high-veracity-risk claimant; contradictions and hedging suggest deception.",
  "reasoning": "Multiple hedging phrases and a direct contradiction about being at the scene significantly lower veracity confidence."
}

Do not include any text outside the JSON object. Do not wrap in markdown fences.
"""

_LLM_USER_TEMPLATE = """\
Analyse the following claimant communication.

CLAIMANT COMMUNICATION:
```
{text}
```
"""


class LLMSentimentAnalyzer:
    """LLM-powered multi-dimensional sentiment analyzer.

    Uses the OpenAI-compatible LM Studio endpoint. Falls back to the
    :class:`RuleBasedSentimentAnalyzer` if the LLM is unavailable or
    returns malformed output.
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self._llm_client = llm_client
        self.tracer = tracer or LangfuseTracer(agent_name="SentimentAgent")
        self._fallback = RuleBasedSentimentAnalyzer()

    def analyze(self, text: str) -> SentimentAssessment:
        """Analyse ``text``. Always returns a :class:`SentimentAssessment`.

        Falls back to the rule-based analyzer if the LLM call fails for
        any reason (timeout, malformed JSON, network error).
        """
        started = time.perf_counter()
        if not text or not text.strip():
            return self._fallback.analyze(text)

        client = self._get_client()
        if client is None:
            result = self._fallback.analyze(text)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        try:
            user_prompt = _LLM_USER_TEMPLATE.format(text=text[:6000])
            resp = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
                timeout=self.config.llm_timeout_sec,
            )
            content = resp.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            parsed = json.loads(content)
            reasoning = parsed.pop("reasoning", None)
            assessment = SentimentAssessment(
                method="llm",
                raw_model_reasoning=reasoning,
                latency_ms=(time.perf_counter() - started) * 1000,
                **parsed,
            )
            return assessment
        except Exception as exc:
            logger.warning(
                "LLM sentiment analysis failed (%s); falling back to rule-based.", exc,
            )
            result = self._fallback.analyze(text)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

    def _get_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        try:
            from openai import OpenAI
            self._llm_client = OpenAI(
                base_url=self.config.lm_studio_base_url,
                api_key=self.config.lm_studio_api_key,
            )
            return self._llm_client
        except Exception as exc:
            logger.debug("OpenAI client construction failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# SentimentOutputParser — strict validation wrapper
# ---------------------------------------------------------------------------
class SentimentOutputParser:
    """Parse and validate LLM output into a :class:`SentimentAssessment`.

    Used by the LLM analyzer internally, but also exposed publicly so
    test suites can validate canned LLM responses without making a
    real LLM call.
    """

    @staticmethod
    def parse(raw: str | dict) -> SentimentAssessment:
        """Parse raw LLM output (string or dict) into a SentimentAssessment.

        Raises ``ValueError`` if the output is malformed.
        """
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM output is not valid JSON: {exc}") from exc
        elif isinstance(raw, dict):
            parsed = raw
        else:
            raise ValueError(f"Expected str or dict, got {type(raw).__name__}")

        # Move "reasoning" to raw_model_reasoning if present
        reasoning = parsed.pop("reasoning", None)
        try:
            return SentimentAssessment(
                method="llm",
                raw_model_reasoning=reasoning,
                **parsed,
            )
        except Exception as exc:
            raise ValueError(f"LLM output failed schema validation: {exc}") from exc


# ---------------------------------------------------------------------------
# SentimentAnalysisEngine — orchestrator with Langfuse tracing
# ---------------------------------------------------------------------------
class SentimentAnalysisEngine:
    """Run the full SP-303 sentiment analysis flow.

    Wraps the LLM analyzer (with rule-based fallback) in a Langfuse
    trace. The trace captures:
    - The raw claimant text (so analysts can audit assessments).
    - The model's reasoning (raw_model_reasoning).
    - The latency per dimension.
    - The method used (llm vs. rule_based).
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
        analyzer: Optional[LLMSentimentAnalyzer] = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.tracer = tracer or LangfuseTracer(agent_name="SentimentAgent")
        self.analyzer = analyzer or LLMSentimentAnalyzer(
            config=self.config, llm_client=llm_client, tracer=self.tracer,
        )

    def analyze(self, text: str, *, claim_id: Optional[str] = None) -> SentimentAssessment:
        """Analyse ``text``. Returns a :class:`SentimentAssessment`."""
        with self.tracer.trace(
            "sentiment_analysis",
            metadata={"claim_id": claim_id or "unknown", "agent.name": "SentimentAgent"},
            tags=["SentimentAgent", "analysis"],
        ):
            assessment = self.analyzer.analyze(text)
            logger.info(
                "SentimentAnalysis: claim_id=%s urgency=%s(%.2f) "
                "emotion=%s(%.2f) veracity=%.2f method=%s",
                claim_id, assessment.urgency_level, assessment.urgency_confidence,
                assessment.emotional_state, assessment.emotional_confidence,
                assessment.veracity_confidence, assessment.method,
            )
            return assessment


# ---------------------------------------------------------------------------
# Labeled evaluation dataset — used by the SP-303 test suite to verify
# the rule-based analyzer meets the AC confidence thresholds.
# ---------------------------------------------------------------------------
# Each entry: (text, expected_urgency, expected_emotional_state,
#              min_veracity_confidence_or_max_if_low)
LABELED_DATASET: list[tuple[str, str, str, float, str]] = [
    # (text, urgency, emotion, veracity_conf_direction, description)
    # High urgency
    ("Please call me back ASAP! The roof is still leaking and I can't wait!",
     "high", "anxious", 0.80, "high urgency + anxious"),

    ("This is an EMERGENCY! I need someone here IMMEDIATELY!",
     "high", "angry", 0.75, "high urgency + angry"),

    ("I'm trapped in my car after an accident! Please help right away!",
     "high", "anxious", 0.85, "life-threatening urgency"),

    # Medium urgency
    ("Hi, I wanted to follow up on my claim. Could someone call me back soon?",
     "medium", "calm", 0.90, "polite follow-up"),

    ("It's been a week and I haven't heard back. Can I get an update?",
     "medium", "frustrated", 0.85, "mild frustration + follow-up"),

    # Low urgency
    ("Thank you for handling my claim. Take your time — I'm in no rush.",
     "low", "calm", 0.95, "calm cooperative"),

    ("I appreciate your help. Please let me know whenever you have an update.",
     "low", "calm", 0.95, "polite patient"),

    # Angry
    ("This is UNACCEPTABLE! I'm furious with your company. I'm calling my attorney!",
     "high", "angry", 0.85, "angry + attorney threat"),

    ("Your incompetence is outrageous. I'm filing a BBB complaint today.",
     "medium", "angry", 0.80, "BBB threat"),

    # Frustrated
    ("I'm so frustrated with this process. I've called five times already.",
     "medium", "frustrated", 0.85, "frustrated repeat caller"),

    ("I'm sick of the runaround. Enough is enough.",
     "medium", "frustrated", 0.80, "fed up"),

    # Anxious
    ("I'm worried about the damage getting worse. Please help.",
     "medium", "anxious", 0.90, "worried"),

    ("I'm scared that my claim will be denied. I don't know what to do.",
     "medium", "anxious", 0.85, "scared uncertain"),

    # Veracity issues
    ("I think the accident was maybe on Tuesday? I'm not really sure. "
     "I was at the scene but I also wasn't there.",
     "low", "anxious", 0.50, "contradiction about presence"),

    ("I don't recall exactly what happened. Let's move on. "
     "I'd rather not say more.",
     "low", "anxious", 0.40, "evasion"),

    ("I think maybe possibly the damage was sort of around $5,000? "
     "I guess. I suppose.",
     "low", "anxious", 0.55, "heavy hedging"),

    # Truthful + detailed
    ("The accident happened on March 14, 2026 at 3:30 PM at the intersection "
     "of Main St and Oak Ave. The other driver ran a red light and rear-ended "
     "my vehicle. I have photos and a police report.",
     "medium", "calm", 0.95, "detailed factual account"),

    ("Thank you for taking my claim. The water heater burst on June 2nd "
     "and flooded the basement. I have a contractor's estimate for $3,200. "
     "Please let me know what other documentation you need.",
     "low", "calm", 0.95, "cooperative detailed"),
]
