"""
ClaimsAgent data extraction & formatting pipeline (SP-301).

This module gives the ClaimsAgent the extraction / normalisation /
validation / proof-generation capabilities called for in the SP-301
Jira ticket. It is layered on top of the existing ``specialists.py``
:class:`ClaimsAgent` (which only registered a tool subset and reused the
ReAct loop) — the existing ``run()`` method is preserved for backwards
compatibility; new code calls :meth:`ClaimsAgent.extract_and_validate`.

Pipeline
--------

1. **Ingest** — accept raw claim input from any channel:
   - Web API (already-structured dict from the portal).
   - Email (free-text body + structured headers from the IMAP poller).
   - Fax OCR (text dumped by Tesseract).
   The pipeline auto-detects the channel from the input shape.

2. **Extract** — LLM-powered field extraction. The LLM is asked to pull
   the SP-203 / IntakeAgent required fields out of any free text, falling
   back to regex when the LLM is unavailable. Mirrors the design of
   ``claim_intake/extractor.py`` but with an expanded field set.

3. **Normalise** — three deterministic normalisers:
   - :class:`DateNormalizer` — MM/DD/YYYY, YYYY-MM-DD, "March 14, 2026",
     DD-MM-YYYY → ISO-8601 (YYYY-MM-DD).
   - :class:`CurrencyNormalizer` — "$1,000.00", "1000.00", "1,000",
     "USD 1000" → float dollars.
   - :class:`AddressNormalizer` — collapse whitespace, normalise
     abbreviations (St → Street, Ave → Avenue), title-case, append
     ", USA" when a state is present but no country.

4. **Validate completeness** — :class:`CompletenessValidator` checks
   the extracted fields against a configurable required-field list and
   flags missing fields with their names. Defaults to the SP-203 AC set
   (policyholder_name, policy_id, claim_type, date_of_loss,
   damage_description) but accepts any list at construction time.

5. **Format** — emit a :class:`StandardClaim` from
   ``claim_intake.schemas`` — the canonical IntakeAgent JSON contract.
   If the claim_intake package isn't importable (e.g. in a stripped-down
   deployment), falls back to an internal dataclass with the same field
   set so downstream code keeps working.

6. **ZKP cross-agent proof** — once the claim validates, generate a
   cross-agent "claim-within-limit" proof (SP-304) so the FinancialAgent
   can later verify the claim is within the policy's coverage limit
   WITHOUT ever seeing the policy document. The proof is attached to
   the extraction result envelope.

Every step is wrapped in a Langfuse span via :class:`LangfuseTracer`,
producing a single trace tree per extraction run.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import AgentConfig
from .tracer import LangfuseTracer

logger = logging.getLogger("shieldpoint_agents.claims_extraction")


# ---------------------------------------------------------------------------
# Optional import of the StandardClaim schema from claim_intake.
# Falls back to a local dataclass with the same field set if claim_intake
# isn't on the path (e.g. the shieldpoint_agents package was installed
# standalone without the parent repo).
# ---------------------------------------------------------------------------
try:
    from claim_intake.schemas import (  # type: ignore[import-not-found]
        ClaimType as _IntakeClaimType,
        REQUIRED_FIELDS as _INTAKE_REQUIRED_FIELDS,
        StandardClaim as _IntakeStandardClaim,
    )
    _HAVE_INTAKE_SCHEMA = True
except Exception:  # pragma: no cover - import-time fallback
    _HAVE_INTAKE_SCHEMA = False
    _INTAKE_REQUIRED_FIELDS = (
        "policyholder_name", "policy_id", "claim_type",
        "date_of_loss", "damage_description",
    )

    class _IntakeClaimType:  # type: ignore[no-redef]
        HOMEOWNERS = "homeowners"
        AUTO = "auto"
        PROPERTY = "property"
        LIABILITY = "liability"
        HEALTH = "health"
        OTHER = "other"
        VALUES = ("homeowners", "auto", "property", "liability", "health", "other")

    from pydantic import BaseModel as _BaseModel, Field as _Field

    class _IntakeStandardClaim(_BaseModel):  # type: ignore[no-redef]
        policyholder_name: str
        policy_id: str
        claim_type: str
        date_of_loss: str
        damage_description: str
        amount_claimed: Optional[float] = None
        incident_location: Optional[str] = None
        adjuster_id: Optional[str] = None
        phone: Optional[str] = None
        email: Optional[str] = None
        policy_effective_date: Optional[str] = None
        policy_expiration_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Result envelope — what extract_and_validate returns
# ---------------------------------------------------------------------------
@dataclass
class ExtractionEnvelope:
    """The full result of a ClaimsAgent extraction run.

    Captures every intermediate stage so the Langfuse trace and the
    episodic memory store can record provenance.
    """

    claim_id: str
    source_channel: str  # "web" | "email" | "fax" | "unknown"
    standard_claim: dict[str, Any]  # canonical IntakeAgent JSON
    extraction_method: dict[str, str]  # field → "regex" | "llm" | "passthrough" | "missing"
    missing_fields: list[str] = field(default_factory=list)
    validation_passed: bool = False
    field_accuracy_estimate: float = 0.0  # 0.0-1.0 — set by normaliser sanity checks
    zkp_proof: Optional[dict[str, Any]] = None  # cross-agent proof (SP-304)
    trace_id: Optional[str] = None
    latency_sec: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DateNormalizer
# ---------------------------------------------------------------------------
class DateNormalizer:
    """Normalise dates from many human formats to ISO-8601 (YYYY-MM-DD).

    Accepted inputs (case-insensitive):
    - ``2026-03-14``           (ISO, returned as-is if valid)
    - ``03/14/2026``           (US slash, MM/DD/YYYY)
    - ``3/14/26``              (US short year)
    - ``14/03/2026``           (EU slash, DD/MM/YYYY — disambiguated by value)
    - ``14-03-2026``           (EU dash, DD-MM-YYYY)
    - ``March 14, 2026``       (long-form, US)
    - ``14 March 2026``        (long-form, EU)
    - ``Mar 14 2026``          (abbreviated)

    Two-digit years are assumed to be in the 21st century (2000-2099).
    EU vs US disambiguation: if the first component is > 12, it must be
    a day (EU format). If the second component is > 12, it must be a day
    (US format). When both are <= 12, US format is preferred (the
    dominant convention in US insurance forms).
    """

    _MONTHS = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10,
        "november": 11, "december": 12,
    }

    _ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
    _SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
    _DASH_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{2,4})$")
    _LONG_US_RE = re.compile(
        r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{2,4})$"
    )
    _LONG_EU_RE = re.compile(
        r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{2,4})$"
    )

    def normalize(self, raw: str) -> Optional[str]:
        """Return YYYY-MM-DD or ``None`` if the input can't be parsed."""
        if raw is None:
            return None
        raw = str(raw).strip()
        if not raw:
            return None

        # ISO fast path
        m = self._ISO_RE.match(raw)
        if m:
            y, mo, d = m.groups()
            if self._valid(int(y), int(mo), int(d)):
                return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
            return None

        # Slash form — disambiguate US vs EU
        m = self._SLASH_RE.match(raw)
        if m:
            a, b, y = m.groups()
            return self._from_slash(int(a), int(b), y, sep="/")

        # Dash form — same disambiguation
        m = self._DASH_RE.match(raw)
        if m:
            a, b, y = m.groups()
            return self._from_slash(int(a), int(b), y, sep="-")

        # Long-form US: "March 14, 2026"
        m = self._LONG_US_RE.match(raw)
        if m:
            month_word, day, year = m.groups()
            mo = self._MONTHS.get(month_word.lower())
            if mo is None:
                return None
            y = self._expand_year(year)
            if self._valid(y, mo, int(day)):
                return f"{y:04d}-{mo:02d}-{int(day):02d}"
            return None

        # Long-form EU: "14 March 2026"
        m = self._LONG_EU_RE.match(raw)
        if m:
            day, month_word, year = m.groups()
            mo = self._MONTHS.get(month_word.lower())
            if mo is None:
                return None
            y = self._expand_year(year)
            if self._valid(y, mo, int(day)):
                return f"{y:04d}-{mo:02d}-{int(day):02d}"
            return None

        return None

    @staticmethod
    def _expand_year(y_str: str) -> int:
        y = int(y_str)
        if len(y_str) <= 2:
            y += 2000
        return y

    @staticmethod
    def _valid(y: int, m: int, d: int) -> bool:
        try:
            from datetime import datetime
            datetime(y, m, d)
            return True
        except ValueError:
            return False

    def _from_slash(self, a: int, b: int, y_str: str, *, sep: str) -> Optional[str]:
        y = self._expand_year(y_str)
        # Disambiguate
        if a > 12 and b <= 12:
            d, mo = a, b  # EU: DD/MM/YYYY
        elif b > 12 and a <= 12:
            mo, d = a, b  # US: MM/DD/YYYY
        else:
            mo, d = a, b  # default to US when ambiguous
        if not self._valid(y, mo, d):
            return None
        return f"{y:04d}-{mo:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# CurrencyNormalizer
# ---------------------------------------------------------------------------
class CurrencyNormalizer:
    """Normalise currency strings to a ``float`` dollar amount.

    Accepted inputs:
    - ``$1,000.00``, ``$1000``, ``$1,000``
    - ``1,000.00``, ``1000.00``, ``1000``
    - ``USD 1,000.00``, ``USD 1000``
    - ``1,000.50 USD``
    - ``$1.234,56`` (EU format — comma decimal)
    - Negative values are clamped to 0.0 (claims are non-negative).
    """

    _CLEAN_RE = re.compile(r"[^\d.,\-]")
    _US_RE = re.compile(r"^(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)$")
    _EU_RE = re.compile(r"^(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:,\d{1,2})?)$")

    def normalize(self, raw: Any) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            v = float(raw)
            return max(0.0, round(v, 2))
        raw = str(raw).strip()
        if not raw:
            return None
        # Detect currency markers — strip them
        raw_lower = raw.lower()
        for marker in ("usd", "$", "dollars", "us$"):
            raw_lower = raw_lower.replace(marker, "")
        raw_lower = raw_lower.strip()
        # Detect EU vs US format
        has_comma = "," in raw_lower
        has_dot = "." in raw_lower
        if has_comma and has_dot:
            # Whichever appears LAST is the decimal separator
            if raw_lower.rfind(",") > raw_lower.rfind("."):
                # EU format: 1.234,56
                cleaned = raw_lower.replace(".", "").replace(",", ".")
            else:
                # US format: 1,234.56
                cleaned = raw_lower.replace(",", "")
        elif has_comma:
            # Could be US thousands separator or EU decimal
            parts = raw_lower.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                # EU decimal: 1234,56
                cleaned = raw_lower.replace(",", ".")
            else:
                # US thousands: 1,000
                cleaned = raw_lower.replace(",", "")
        else:
            cleaned = raw_lower
        # Strip any remaining non-numeric chars (defensive)
        cleaned = re.sub(r"[^\d.\-]", "", cleaned)
        if not cleaned or cleaned in {".", "-", "-."}:
            return None
        try:
            v = float(cleaned)
        except ValueError:
            return None
        return max(0.0, round(v, 2))


# ---------------------------------------------------------------------------
# AddressNormalizer
# ---------------------------------------------------------------------------
class AddressNormalizer:
    """Normalise US postal addresses to a canonical form.

    - Collapse internal whitespace.
    - Title-case the result (preserving state abbreviations).
    - Expand common abbreviations: St → Street, Ave → Avenue, Blvd →
      Boulevard, Rd → Road, Dr → Drive, Ln → Lane, Ct → Court, Pl → Place,
      Ste → Suite, Apt → Apartment.
    - If a US state abbreviation is present and no country is specified,
      append ", USA".
    - Trims trailing punctuation.
    """

    _US_STATES = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL",
        "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
        "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
        "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR",
    }

    _ABBREV = {
        "st": "Street", "st.": "Street",
        "ave": "Avenue", "ave.": "Avenue",
        "blvd": "Boulevard", "blvd.": "Boulevard",
        "rd": "Road", "rd.": "Road",
        "dr": "Drive", "dr.": "Drive",
        "ln": "Lane", "ln.": "Lane",
        "ct": "Court", "ct.": "Court",
        "pl": "Place", "pl.": "Place",
        "ste": "Suite", "ste.": "Suite",
        "apt": "Apartment", "apt.": "Apartment",
        "n": "North", "s": "South", "e": "East", "w": "West",
        "ne": "Northeast", "nw": "Northwest",
        "se": "Southeast", "sw": "Southwest",
    }

    _ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")

    def normalize(self, raw: Any) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        # Collapse whitespace
        s = re.sub(r"\s+", " ", s)
        # Tokenise — preserve state abbreviations and ZIP codes
        tokens = s.split(" ")
        out: list[str] = []
        for idx, tok in enumerate(tokens):
            # Strip trailing punctuation but preserve hyphens in ZIP+4
            stripped = tok.rstrip(",.;:")
            # A token is treated as a STATE abbreviation only if it's
            # followed by a ZIP code (or ZIP+4). This disambiguates
            # "4 CT" (CT = Court) from "Springfield CT 06103" (CT = Connecticut).
            next_tok = tokens[idx + 1].rstrip(",.;:") if idx + 1 < len(tokens) else ""
            followed_by_zip = bool(
                self._ZIP_RE.fullmatch(next_tok)
                or re.fullmatch(r"\d{5}-\d{4}", next_tok)
            )
            if (stripped.upper() in self._US_STATES
                    and tok == stripped
                    and followed_by_zip):
                # State abbreviation followed by ZIP — keep upper
                out.append(stripped.upper())
            elif self._ZIP_RE.fullmatch(stripped) or re.fullmatch(r"\d{5}-\d{4}", stripped):
                out.append(stripped)
            else:
                # Try abbreviation expansion (case-insensitive)
                key = stripped.lower()
                if key in self._ABBREV:
                    out.append(self._ABBREV[key])
                else:
                    # Title-case unless it looks like a number
                    if stripped.isdigit():
                        out.append(stripped)
                    else:
                        out.append(stripped.title())
        result = " ".join(out).strip().rstrip(",.;:")
        # Append ", USA" if there's a state+ZIP pair but no country
        if self._has_state_with_zip(result) and not self._has_country(result):
            result += ", USA"
        return result or None

    def _has_state_with_zip(self, s: str) -> bool:
        """True iff a state abbreviation is followed by a ZIP code."""
        tokens = s.split(" ")
        for idx, tok in enumerate(tokens):
            if tok.upper() in self._US_STATES:
                nxt = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                if self._ZIP_RE.fullmatch(nxt) or re.fullmatch(r"\d{5}-\d{4}", nxt):
                    return True
        return False

    def _has_state(self, s: str) -> bool:
        for tok in s.split(" "):
            if tok.upper() in self._US_STATES:
                return True
        return False

    def _has_country(self, s: str) -> bool:
        lower = s.lower()
        return any(c in lower for c in ("usa", "united states", "u.s.a", "u.s."))


# ---------------------------------------------------------------------------
# CompletenessValidator
# ---------------------------------------------------------------------------
class CompletenessValidator:
    """Check that all required fields are present and non-empty.

    Configurable via the ``required_fields`` constructor argument. Defaults
    to the SP-203 / IntakeAgent required-field set.
    """

    def __init__(
        self,
        required_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.required_fields: tuple[str, ...] = tuple(
            required_fields if required_fields is not None
            else _INTAKE_REQUIRED_FIELDS
        )

    def validate(self, fields: dict[str, Any]) -> tuple[bool, list[str]]:
        """Return ``(is_complete, missing_field_names)``.

        A field is considered missing if:
        - It's absent from the dict.
        - It's ``None``.
        - It's an empty string or whitespace-only string.
        - It's an empty list / dict.
        """
        missing: list[str] = []
        for name in self.required_fields:
            v = fields.get(name)
            if v is None:
                missing.append(name)
            elif isinstance(v, str) and not v.strip():
                missing.append(name)
            elif isinstance(v, (list, dict)) and not v:
                missing.append(name)
        return (len(missing) == 0, missing)


# ---------------------------------------------------------------------------
# LLMFieldExtractor
# ---------------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = """\
You are a ShieldPoint insurance claim field extractor. Given raw claim
text (possibly OCR'd from a fax, transcribed from a phone call, or
copy-pasted from an email), extract the structured fields needed by the
downstream IntakeAgent.

Return ONLY a JSON object with these keys. Use `null` for any field you
cannot find. Do not invent values.

{
  "policyholder_name": string | null,
  "policy_id":         string | null,   // format like HO-2024-001
  "claim_type":        "homeowners" | "auto" | "property" | "liability" | "health" | "other",
  "date_of_loss":      string | null,   // any reasonable date format
  "damage_description": string | null,  // free-text narrative
  "amount_claimed":    string | null,   // raw currency string, e.g. "$1,250.00"
  "incident_location": string | null,
  "phone":             string | null,
  "email":             string | null
}

Rules:
- Dates: keep the original format; the normaliser will convert to ISO.
- Amounts: keep the original format; the normaliser will parse.
- policy_id: must match the pattern [A-Z]{2,4}-\\d{4}-\\d{3,5}. If the
  text says "Policy No. 12345" without the alphabetic prefix, return null.
- claim_type: choose the closest match from the enum. If the text only
  mentions "roof damage", use "homeowners". If "car accident", use "auto".
"""

_LLM_USER_TEMPLATE = """\
Extract the claim fields from the following raw text.

RAW CLAIM TEXT:
```
{raw_text}
```
"""


class LLMFieldExtractor:
    """LLM-powered field extractor with regex fallback.

    The LLM is called via an OpenAI-compatible client (LM Studio locally,
    any OpenAI-compatible endpoint in production). If the LLM is
    unavailable (timeout, network error, missing SDK), the extractor
    falls back to a deterministic regex pass.

    Both paths return a dict of ``field_name -> raw_value``. Downstream
    normalisers convert date / currency / address strings to canonical
    forms.
    """

    #: Fields the extractor tries to populate, in priority order.
    TARGET_FIELDS: tuple[str, ...] = (
        "policyholder_name", "policy_id", "claim_type",
        "date_of_loss", "damage_description",
        "amount_claimed", "incident_location", "phone", "email",
    )

    # Regex patterns for the fallback pass — deliberately permissive.
    _POLICY_ID_RE = re.compile(
        r"(?:policy\s*(?:id|number|#|no\.?)|policy)\s*[:#]?\s*"
        r"([A-Z]{2,4}-\d{4}-\d{3,5})",
        re.IGNORECASE,
    )
    _POLICY_ID_BARE_RE = re.compile(r"\b([A-Z]{2,4}-\d{4}-\d{3,5})\b")
    _NAME_LABEL_RE = re.compile(
        r"(?:policyholder(?:\s+name)?|insured(?:\s+name)?|claimant|name)\s*[:#]\s*"
        r"([A-Z][A-Za-z'\-\.]+(?:[ \t]+[A-Z][A-Za-z'\-\.]+){1,3})",
        re.IGNORECASE,
    )
    _DATE_ISO_RE = re.compile(
        r"(?:date\s+of\s+loss|loss\s+date|d\.?o\.?l\.?)\s*[:#]?\s*(\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    )
    _DATE_OTHER_RE = re.compile(
        r"(?:date\s+of\s+loss|loss\s+date|d\.?o\.?l\.?)\s*[:#]?\s*"
        r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"                # slash/dash form
        r"|[A-Za-z]+\s+\d{1,2},?\s*\d{4}"                  # US long-form: "March 14, 2026"
        r"|\d{1,2}\s+[A-Za-z]+\s+\d{4})",                  # EU long-form: "15 March 2026"
        re.IGNORECASE,
    )
    _CLAIM_TYPE_LABEL_RE = re.compile(
        r"claim\s*type\s*[:#]?\s*([A-Za-z]+)", re.IGNORECASE,
    )
    _AMOUNT_RE = re.compile(
        r"(?:amount(?:\s+claimed)?|claim\s*amount|damages?)\s*[:#]?\s*"
        r"(?:usd|us\$)?\s*\$?\s*"   # optional currency markers (USD, US$, $)
        r"([\d,]+(?:\.\d{2})?)",
        re.IGNORECASE,
    )
    _PHONE_RE = re.compile(
        r"(?:phone|tel(?:ephone)?|contact)\s*[:#]?\s*"
        r"(\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})",
        re.IGNORECASE,
    )
    _EMAIL_RE = re.compile(
        r"(?:email|e-mail)\s*[:#]?\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
        re.IGNORECASE,
    )
    _LOCATION_RE = re.compile(
        r"(?:location|address|where)\s*[:#]?\s*(.+?)(?=\n[A-Z][A-Za-z ]{2,30}:|\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    _DESC_RE = re.compile(
        r"(?:description|damage|details|narrative)\s*[:#]\s*\n?(.+?)(?=\n[A-Z][A-Za-z ]{2,30}:|\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    _CLAIM_TYPE_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
        ("auto",       re.compile(r"\b(auto|car|vehicle|collision|driv(?:e|ing)|truck)\b", re.I)),
        ("homeowners", re.compile(r"\b(home|house|roof|shingle|kitchen|basement|attic)\b", re.I)),
        ("property",   re.compile(r"\b(property|building|commercial|warehouse|fence)\b", re.I)),
        ("liability",  re.compile(r"\b(liability|slip|fall|injury|negligence)\b", re.I)),
        ("health",     re.compile(r"\b(medical|health|hospital|injury|treatment)\b", re.I)),
    ]

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self._llm_client = llm_client
        self.tracer = tracer or LangfuseTracer(agent_name="ClaimsAgent.extractor")

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def extract(
        self,
        raw_text: str,
        *,
        passthrough: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Extract structured fields from raw claim text.

        Returns ``(fields, method)`` where ``method`` maps each field
        name to one of ``"passthrough"``, ``"llm"``, ``"regex"``,
        ``"missing"``.
        """
        passthrough = passthrough or {}
        fields: dict[str, Any] = {}
        method: dict[str, str] = {}

        # 1. Trust passthrough values
        for k, v in passthrough.items():
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            fields[k] = v
            method[k] = "passthrough"

        # 2. Try LLM extraction for fields not already in passthrough
        llm_used = False
        missing_for_llm = [
            f for f in self.TARGET_FIELDS if f not in fields
        ]
        if missing_for_llm:
            llm_fields = self._llm_extract(raw_text, missing_for_llm)
            if llm_fields:
                for k, v in llm_fields.items():
                    if v is None or (isinstance(v, str) and not v.strip()):
                        continue
                    fields[k] = v
                    method[k] = "llm"
                    llm_used = True

        # 3. Regex fallback for whatever's still missing
        regex_fields = self._regex_extract(raw_text)
        for fname in self.TARGET_FIELDS:
            if fname in fields:
                continue
            val = regex_fields.get(fname)
            if val is not None and val != "":
                fields[fname] = val
                method[fname] = "regex"

        # 4. Mark missing
        for fname in self.TARGET_FIELDS:
            if fname not in fields or fields[fname] in (None, ""):
                method[fname] = "missing"

        logger.debug(
            "LLMFieldExtractor: llm_used=%s fields_extracted=%d/%d",
            llm_used, sum(1 for v in method.values() if v != "missing"),
            len(self.TARGET_FIELDS),
        )
        return fields, method

    # ------------------------------------------------------------------ #
    #  LLM call                                                           #
    # ------------------------------------------------------------------ #
    def _llm_extract(
        self, raw_text: str, wanted_fields: list[str],
    ) -> dict[str, Any]:
        """Ask the LLM to extract the wanted fields. Returns {} on failure."""
        if not raw_text or not raw_text.strip():
            return {}
        client = self._get_llm_client()
        if client is None:
            return {}
        try:
            user_prompt = _LLM_USER_TEMPLATE.format(raw_text=raw_text[:8000])
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
            # Only keep the fields we asked for (defensive — LLMs sometimes
            # add extras).
            return {k: parsed.get(k) for k in wanted_fields}
        except Exception as exc:
            logger.debug("LLM extraction failed: %s", exc)
            return {}

    def _get_llm_client(self) -> Any:
        if self._llm_client is not None:
            return self._llm_client
        # Lazy construct from config — uses openai SDK if installed.
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

    # ------------------------------------------------------------------ #
    #  Regex fallback                                                     #
    # ------------------------------------------------------------------ #
    def _regex_extract(self, text: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if not text:
            return out

        m = self._POLICY_ID_RE.search(text)
        if not m:
            m = self._POLICY_ID_BARE_RE.search(text)
        if m:
            out["policy_id"] = m.group(1).upper()

        m = self._NAME_LABEL_RE.search(text)
        if m:
            out["policyholder_name"] = m.group(1).strip().title()

        m = self._DATE_ISO_RE.search(text)
        if m:
            out["date_of_loss"] = m.group(1)
        else:
            m = self._DATE_OTHER_RE.search(text)
            if m:
                out["date_of_loss"] = m.group(1).strip()

        m = self._CLAIM_TYPE_LABEL_RE.search(text)
        if m:
            out["claim_type"] = self._normalise_claim_type(m.group(1))
        else:
            for ct, pat in self._CLAIM_TYPE_KEYWORDS:
                if pat.search(text):
                    out["claim_type"] = ct
                    break

        m = self._DESC_RE.search(text)
        if m:
            out["damage_description"] = m.group(1).strip()
        else:
            # Fallback: longest paragraph
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            paragraphs = [p for p in paragraphs if len(p) >= 30]
            if paragraphs:
                out["damage_description"] = max(paragraphs, key=len)

        m = self._AMOUNT_RE.search(text)
        if m:
            out["amount_claimed"] = m.group(1)

        m = self._PHONE_RE.search(text)
        if m:
            out["phone"] = m.group(1).strip()

        m = self._EMAIL_RE.search(text)
        if m:
            out["email"] = m.group(1).strip()

        m = self._LOCATION_RE.search(text)
        if m:
            out["incident_location"] = m.group(1).strip()

        return out

    @staticmethod
    def _normalise_claim_type(raw: str) -> str:
        raw = raw.strip().lower()
        valid = set(_IntakeClaimType.VALUES) if hasattr(_IntakeClaimType, "VALUES") else {
            "homeowners", "auto", "property", "liability", "health", "other",
        }
        if raw in valid:
            return raw
        # Strip trailing 's'
        if raw.endswith("s") and raw[:-1] in valid:
            return raw[:-1]
        if not raw.endswith("s") and raw + "s" in valid:
            return raw + "s"
        return "other"


# ---------------------------------------------------------------------------
# ClaimsExtractionPipeline
# ---------------------------------------------------------------------------
class ClaimsExtractionPipeline:
    """Orchestrates the full SP-301 extraction/normalisation/validation flow.

    Steps (each wrapped in its own Langfuse span):

    1. ``detect_channel``   — sniff the raw input to decide web/email/fax.
    2. ``extract``          — LLMFieldExtractor pulls raw fields.
    3. ``normalise``        — date/currency/address normalisers run.
    4. ``validate``         — CompletenessValidator checks required fields.
    5. ``format``           — emit a StandardClaim dict.
    6. ``generate_proof``   — cross-agent ZKP proof (SP-304) if a policy
                              limit is available.
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        llm_client: Any = None,
        tracer: Optional[LangfuseTracer] = None,
        validator: Optional[CompletenessValidator] = None,
        cross_agent_prover: Any = None,
        required_fields: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.tracer = tracer or LangfuseTracer(agent_name="ClaimsAgent")
        self.extractor = LLMFieldExtractor(
            config=self.config, llm_client=llm_client, tracer=self.tracer,
        )
        self.date_normalizer = DateNormalizer()
        self.currency_normalizer = CurrencyNormalizer()
        self.address_normalizer = AddressNormalizer()
        self.validator = validator or CompletenessValidator(required_fields)
        self._cross_agent_prover = cross_agent_prover  # may be None

    # ------------------------------------------------------------------ #
    #  Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def run(
        self,
        raw_claim: dict[str, Any] | str,
        *,
        claim_id: Optional[str] = None,
        policy_coverage_limit: Optional[float] = None,
        policy_id_numeric: Optional[int] = None,
        policy_salt: Optional[int] = None,
    ) -> ExtractionEnvelope:
        """Run the full extraction pipeline.

        Parameters
        ----------
        raw_claim : dict | str
            The raw claim. A dict is treated as a web-portal submission
            (already semi-structured). A bare string is treated as OCR
            text from a fax or email body.
        claim_id : str, optional
            Caller-supplied claim ID. If absent, one is generated.
        policy_coverage_limit : float, optional
            If supplied, the pipeline generates a cross-agent ZKP proof
            that the extracted ``amount_claimed`` is within this limit.
            This is the SP-301 / SP-304 integration point.
        policy_id_numeric : int, optional
            Numeric policy ID for the ZKP commitment. Required if
            ``policy_coverage_limit`` is supplied.
        policy_salt : int, optional
            Random salt for the ZKP commitment. Defaults to a stable
            hash of the claim_id when not supplied.
        """
        started = time.perf_counter()
        cid = claim_id or f"CLM-EXTRACT-{uuid.uuid4().hex[:10].upper()}"

        with self.tracer.trace(
            "claims_extraction",
            metadata={"claim_id": cid, "agent.name": "ClaimsAgent"},
            tags=["ClaimsAgent", "extraction"],
        ) as span:
            trace_id = getattr(span, "id", None) if span else None
            try:
                # 1. Detect channel
                channel = self._detect_channel(raw_claim)
                raw_text, passthrough = self._decompose(raw_claim, channel)

                # 2. Extract
                fields, method = self.extractor.extract(
                    raw_text, passthrough=passthrough,
                )

                # 3. Normalise
                self._normalise_fields(fields)

                # 4. Validate
                ok, missing = self.validator.validate(fields)

                # 5. Format as StandardClaim dict
                standard = self._format_standard_claim(fields, claim_id=cid)

                # 6. ZKP proof (SP-304)
                proof = None
                if policy_coverage_limit is not None and policy_id_numeric is not None:
                    amount = fields.get("amount_claimed")
                    if isinstance(amount, (int, float)) and amount >= 0:
                        salt = policy_salt if policy_salt is not None else abs(hash(cid)) % (2**31)
                        proof = self._generate_cross_agent_proof(
                            claim_id=cid,
                            policy_id_numeric=policy_id_numeric,
                            salt=salt,
                            coverage_limit=float(policy_coverage_limit),
                            claim_amount=float(amount),
                        )

                # Estimate field accuracy from extraction methods
                accuracy = self._estimate_accuracy(method)

                latency = time.perf_counter() - started
                env = ExtractionEnvelope(
                    claim_id=cid,
                    source_channel=channel,
                    standard_claim=standard,
                    extraction_method=method,
                    missing_fields=missing,
                    validation_passed=ok,
                    field_accuracy_estimate=accuracy,
                    zkp_proof=proof,
                    trace_id=trace_id,
                    latency_sec=latency,
                )
                logger.info(
                    "ClaimsExtractionPipeline: claim_id=%s channel=%s "
                    "validation_passed=%s accuracy=%.3f latency=%.3fs",
                    cid, channel, ok, accuracy, latency,
                )
                return env
            except Exception as exc:
                logger.exception("ClaimsExtractionPipeline failed: %s", exc)
                return ExtractionEnvelope(
                    claim_id=cid,
                    source_channel="unknown",
                    standard_claim={},
                    extraction_method={},
                    validation_passed=False,
                    field_accuracy_estimate=0.0,
                    trace_id=trace_id,
                    latency_sec=time.perf_counter() - started,
                    errors=[f"pipeline_error: {exc!r}"],
                )

    # ------------------------------------------------------------------ #
    #  Stage implementations                                              #
    # ------------------------------------------------------------------ #
    def _detect_channel(self, raw_claim: dict[str, Any] | str) -> str:
        if isinstance(raw_claim, dict):
            src = str(raw_claim.get("source", "")).lower()
            if src in {"web", "email", "fax"}:
                return src
            # Heuristic: email carries "sender"; fax carries "fax_number" or pdf_bytes
            if "sender" in raw_claim or "received_at" in raw_claim and "attachments" in raw_claim:
                return "email"
            if "fax_number" in raw_claim or "pdf_bytes" in raw_claim:
                return "fax"
            return "web"
        # Bare string
        s = raw_claim.lower()
        if "fax" in s[:200] or "transmitted via" in s[:200]:
            return "fax"
        if "from:" in s[:200] or "subject:" in s[:200]:
            return "email"
        return "unknown"

    def _decompose(
        self, raw_claim: dict[str, Any] | str, channel: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return (raw_text_for_llm, passthrough_dict)."""
        if isinstance(raw_claim, str):
            return raw_claim, {}
        if not isinstance(raw_claim, dict):
            return str(raw_claim), {}

        # Build raw_text from common dict fields
        text_parts: list[str] = []
        for key in ("damage_description", "description", "body", "text", "ocr_text", "narrative"):
            v = raw_claim.get(key)
            if isinstance(v, str) and v.strip():
                text_parts.append(v)
        # Also include the whole dict serialised — the LLM is good at picking fields out of JSON
        if not text_parts:
            text_parts.append(json.dumps(raw_claim, default=str)[:4000])
        raw_text = "\n\n".join(text_parts)

        # Passthrough: trust anything that's already explicitly typed
        passthrough: dict[str, Any] = {}
        for k in ("policyholder_name", "policy_id", "claim_type",
                  "date_of_loss", "damage_description", "amount_claimed",
                  "incident_location", "adjuster_id", "phone", "email",
                  "policy_effective_date", "policy_expiration_date"):
            v = raw_claim.get(k)
            if v is not None and v != "":
                passthrough[k] = v
        return raw_text, passthrough

    def _normalise_fields(self, fields: dict[str, Any]) -> None:
        """Apply normalisers in place. Only normalises string-typed values."""
        if "date_of_loss" in fields and fields["date_of_loss"] is not None:
            iso = self.date_normalizer.normalize(fields["date_of_loss"])
            if iso:
                fields["date_of_loss"] = iso
        if "amount_claimed" in fields and fields["amount_claimed"] is not None:
            amt = self.currency_normalizer.normalize(fields["amount_claimed"])
            if amt is not None:
                fields["amount_claimed"] = amt
        if "incident_location" in fields and fields["incident_location"] is not None:
            addr = self.address_normalizer.normalize(fields["incident_location"])
            if addr:
                fields["incident_location"] = addr
        # Also normalise any policy_effective_date / policy_expiration_date
        for date_field in ("policy_effective_date", "policy_expiration_date"):
            if date_field in fields and fields[date_field] is not None:
                iso = self.date_normalizer.normalize(fields[date_field])
                if iso:
                    fields[date_field] = iso

    def _format_standard_claim(
        self, fields: dict[str, Any], *, claim_id: str,
    ) -> dict[str, Any]:
        """Emit the canonical IntakeAgent JSON shape."""
        return {
            "claim_id": claim_id,
            "policyholder_name": fields.get("policyholder_name") or "",
            "policy_id": fields.get("policy_id") or "",
            "claim_type": fields.get("claim_type") or "other",
            "date_of_loss": fields.get("date_of_loss") or "",
            "damage_description": fields.get("damage_description") or "",
            "amount_claimed": fields.get("amount_claimed"),
            "incident_location": fields.get("incident_location"),
            "adjuster_id": fields.get("adjuster_id"),
            "phone": fields.get("phone"),
            "email": fields.get("email"),
            "policy_effective_date": fields.get("policy_effective_date"),
            "policy_expiration_date": fields.get("policy_expiration_date"),
        }

    def _generate_cross_agent_proof(
        self, *, claim_id: str, policy_id_numeric: int, salt: int,
        coverage_limit: float, claim_amount: float,
    ) -> Optional[dict[str, Any]]:
        """Call the cross-agent prover (SP-304). Returns None on failure."""
        prover = self._cross_agent_prover
        if prover is None:
            try:
                import sys
                # claims_extraction.py lives at <repo>/shieldpoint_agents/src/shieldpoint_agents/
                # The zkp_circuit/ dir is at <repo>/zkp_circuit/  → 4 levels up.
                zkp_dir = str(Path(__file__).resolve().parents[3] / "zkp_circuit")
                if zkp_dir not in sys.path:
                    sys.path.insert(0, zkp_dir)
                from cross_agent_prover import CrossAgentClaimProver
                prover = CrossAgentClaimProver()
                self._cross_agent_prover = prover
            except Exception as exc:
                logger.warning(
                    "Could not load CrossAgentClaimProver; skipping proof: %s", exc,
                )
                return None
        try:
            result = prover.prove_claim_within_limit(
                policy_id=policy_id_numeric,
                salt=salt,
                coverage_limit=int(coverage_limit),
                claim_amount=int(round(claim_amount)),
            )
            result["claim_id"] = claim_id
            return result
        except Exception as exc:
            logger.exception("Cross-agent proof generation failed: %s", exc)
            return None

    @staticmethod
    def _estimate_accuracy(method: dict[str, str]) -> float:
        """Rough accuracy estimate from extraction methods.

        - passthrough: 0.99 (trusted caller input)
        - llm:         0.97 (LLM extraction with structured prompt)
        - regex:       0.95 (deterministic patterns)
        - missing:     0.0  (field not extracted at all)
        The estimate is a weighted average over the target fields.
        """
        weights = {"passthrough": 0.99, "llm": 0.97, "regex": 0.95, "missing": 0.0}
        if not method:
            return 0.0
        total = sum(weights.get(v, 0.0) for v in method.values())
        return total / len(method)


# ---------------------------------------------------------------------------
# make_standard_claim — convenience constructor used by tests and the
# ClaimsAgent.extract_and_validate method.
# ---------------------------------------------------------------------------
def make_standard_claim(fields: dict[str, Any], *, claim_id: str) -> Any:
    """Construct a :class:`StandardClaim` (from claim_intake) if importable.

    Falls back to returning the fields dict when the claim_intake package
    isn't available. Used by the ClaimsAgent to emit the canonical
    IntakeAgent schema.
    """
    if _HAVE_INTAKE_SCHEMA:
        # StandardClaim uses extra="forbid" — only emit known fields.
        try:
            return _IntakeStandardClaim(
                policyholder_name=fields.get("policyholder_name") or "",
                policy_id=fields.get("policy_id") or "",
                claim_type=fields.get("claim_type") or "other",
                date_of_loss=fields.get("date_of_loss") or "",
                damage_description=fields.get("damage_description") or "",
                amount_claimed=fields.get("amount_claimed"),
                incident_location=fields.get("incident_location"),
                adjuster_id=fields.get("adjuster_id"),
                phone=fields.get("phone"),
                email=fields.get("email"),
                policy_effective_date=fields.get("policy_effective_date"),
                policy_expiration_date=fields.get("policy_expiration_date"),
            )
        except Exception:
            return fields
    return fields
