"""
Field extraction pipeline for OCR'd claim text (SP-203).

Two-tier extraction:

1. **Regex pass** — fast, deterministic, and runs first. Covers the
   well-structured parts of a fax/email claim form (e.g. "Policy ID:
   HO-2024-001"). Each field has its own targeted pattern.

2. **LLM-assisted pass** — runs only on fields the regex missed. Calls an
   OpenAI-compatible endpoint (LM Studio locally) with a constrained
   prompt asking it to extract the missing field. Falls back gracefully
   if the LLM is unreachable — the field stays empty and the validator
   flags it.

Why two tiers? The regex pass alone gives us ~90% coverage on well-formed
faxes. The LLM pass catches the remaining ~10% (handwritten sections OCR'd
poorly, unusual date formats, etc.) without making the LLM a hard dependency.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .config import IntakeConfig
from .schemas import ClaimType

logger = logging.getLogger("claim_intake.extractor")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """The output of the extraction pipeline.

    ``fields`` is a flat dict of field_name → value, where value is a string
    (or None if not found). ``method`` records which strategy populated
    each field — useful for debugging and for the confidence scorer.
    """

    fields: dict[str, Any] = field(default_factory=dict)
    #: field_name → "regex" | "llm" | "passthrough" | "missing"
    method: dict[str, str] = field(default_factory=dict)
    #: True iff the LLM was actually consulted (vs regex-only).
    llm_used: bool = False
    #: Number of fields that could not be extracted.
    missing_count: int = 0

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Policy IDs come in many shapes — we match the common ones:
#   HO-2024-001      (homeowners)
#   AU-2024-015      (auto)
#   POL-2024-12345   (generic)
#   GL-2024-001      (general liability)
_POLICY_ID_RE = re.compile(
    r"(?:policy\s*(?:id|number|#|no\.?)|policy)\s*[:#]?\s*"
    r"([A-Z]{2,4}-\d{4}-\d{3,5})",
    re.IGNORECASE,
)

# Free-standing policy IDs (no "Policy ID:" prefix) — common on fax forms
# where the field label is on a different line.
_POLICY_ID_BARE_RE = re.compile(
    r"\b([A-Z]{2,4}-\d{4}-\d{3,5})\b"
)

# Policyholder name — appears after "Name:", "Insured:", "Claimant:", etc.
# Capture up to the end of the line (use [ \t] not \s to avoid matching
# across newlines). Accept optional "Name" suffix on "Policyholder"/"Insured".
_NAME_LABEL_RE = re.compile(
    r"(?:policyholder(?:\s+name)?|insured(?:\s+name)?|claimant|name)\s*[:#]\s*"
    r"([A-Z][A-Za-z'\-\.]+(?:[ \t]+[A-Z][A-Za-z'\-\.]+){1,3})",
    re.IGNORECASE,
)

# Date of loss — accept ISO (2026-03-14) and US (03/14/2026, 3/14/26)
# formats. Normalised to ISO downstream.
_DATE_ISO_RE = re.compile(
    r"(?:date\s+of\s+loss|loss\s+date|d\.?o\.?l\.?)\s*[:#]?\s*"
    r"(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_DATE_US_RE = re.compile(
    r"(?:date\s+of\s+loss|loss\s+date|d\.?o\.?l\.?)\s*[:#]?\s*"
    r"(\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)

# Claim type — try the explicit "Claim Type:" label first, then infer from
# keywords in the description.
_CLAIM_TYPE_LABEL_RE = re.compile(
    r"claim\s*type\s*[:#]?\s*([A-Za-z]+)",
    re.IGNORECASE,
)

# Claim type inference keywords. The first match wins.
_CLAIM_TYPE_KEYWORDS: list[tuple[ClaimType, re.Pattern[str]]] = [
    (ClaimType.AUTO, re.compile(r"\b(auto|car|vehicle|collision|driv(?:e|ing)|truck)\b", re.I)),
    (ClaimType.HOMEOWNERS, re.compile(r"\b(home|house|roof|shingle|kitchen|basement|attic)\b", re.I)),
    (ClaimType.PROPERTY, re.compile(r"\b(property|building|commercial|warehouse|fence)\b", re.I)),
    (ClaimType.LIABILITY, re.compile(r"\b(liability|slip|fall|injury|negligence)\b", re.I)),
    (ClaimType.HEALTH, re.compile(r"\b(medical|health|hospital|injury|treatment)\b", re.I)),
]

# Damage description — capture the longest paragraph after a "Description:"
# label, or the longest paragraph in the document if no label is present.
# Stop at the next "Label:" line OR two newlines OR end of text.
_LABEL_LOOKAHEAD = r"(?=\n[A-Z][A-Za-z ]{2,30}:|\n\n|\Z)"
_DESC_LABEL_RE = re.compile(
    r"(?:description|damage|details|narrative)\s*[:#]\s*\n?(.+?)" + _LABEL_LOOKAHEAD,
    re.IGNORECASE | re.DOTALL,
)

# Amount — dollar-formatted. Accept "Amount:", "Claim Amount:", "Amount Claimed:",
# "Damages:".
_AMOUNT_RE = re.compile(
    r"(?:amount(?:\s+claimed)?|claim\s*amount|damages?)\s*[:#]?\s*\$?\s*"
    r"([\d,]+(?:\.\d{2})?)",
    re.IGNORECASE,
)

# Phone numbers — US-format primarily.
_PHONE_RE = re.compile(
    r"(?:phone|tel(?:ephone)?|contact)\s*[:#]?\s*"
    r"(\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})",
    re.IGNORECASE,
)

# Email — RFC-5322-lite.
_EMAIL_RE = re.compile(
    r"(?:email|e-mail)\s*[:#]?\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)

# Incident location.
_LOCATION_RE = re.compile(
    r"(?:location|address|where)\s*[:#]?\s*(.+?)" + _LABEL_LOOKAHEAD,
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_date(raw: str) -> str:
    """Convert a US-format date (M/D/YYYY) to ISO (YYYY-MM-DD)."""
    raw = raw.strip()
    # Already ISO?
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", raw)
    if not m:
        return raw  # let the validator catch it
    month, day, year = m.groups()
    if len(year) == 2:
        year = "20" + year  # assume 21st century
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _normalise_claim_type(raw: str) -> ClaimType:
    """Map a free-text claim type to the enum, defaulting to OTHER.

    Accepts the exact enum value (``homeowners``), the singular form
    (``homeowner``), and the plural form (``homeownerss``). Anything
    else falls through to keyword inference, then OTHER.
    """
    raw = raw.strip().lower()
    for ct in ClaimType:
        if raw == ct.value:
            return ct
        # Strip trailing 's' for plural-form comparison.
        if ct.value.endswith("s") and raw == ct.value[:-1]:
            return ct
        if not ct.value.endswith("s") and raw == ct.value + "s":
            return ct
    # Try keyword inference as a fallback
    for ct, pattern in _CLAIM_TYPE_KEYWORDS:
        if pattern.search(raw):
            return ct
    return ClaimType.OTHER


def _longest_paragraph(text: str, *, min_len: int = 30) -> str | None:
    """Return the longest paragraph in ``text`` that meets ``min_len``.

    Used as a fallback for damage description when no explicit label is
    present.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paragraphs = [p for p in paragraphs if len(p) >= min_len]
    if not paragraphs:
        return None
    return max(paragraphs, key=len)


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------
def _regex_extract(text: str) -> dict[str, Any]:
    """Run all regex patterns and return the captured values."""
    out: dict[str, Any] = {}

    # Policy ID — try labelled, then bare.
    m = _POLICY_ID_RE.search(text)
    if not m:
        m = _POLICY_ID_BARE_RE.search(text)
    if m:
        out["policy_id"] = m.group(1).upper()

    # Policyholder name
    m = _NAME_LABEL_RE.search(text)
    if m:
        out["policyholder_name"] = m.group(1).strip().title()

    # Date of loss — try ISO, then US.
    m = _DATE_ISO_RE.search(text)
    if m:
        out["date_of_loss"] = m.group(1)
    else:
        m = _DATE_US_RE.search(text)
        if m:
            out["date_of_loss"] = _normalise_date(m.group(1))

    # Claim type
    m = _CLAIM_TYPE_LABEL_RE.search(text)
    if m:
        out["claim_type"] = _normalise_claim_type(m.group(1)).value
    else:
        # Infer from description keywords
        for ct, pattern in _CLAIM_TYPE_KEYWORDS:
            if pattern.search(text):
                out["claim_type"] = ct.value
                break

    # Damage description — labelled, then longest paragraph.
    m = _DESC_LABEL_RE.search(text)
    if m:
        out["damage_description"] = m.group(1).strip()
    else:
        longest = _longest_paragraph(text)
        if longest:
            out["damage_description"] = longest

    # Amount
    m = _AMOUNT_RE.search(text)
    if m:
        try:
            out["amount_claimed"] = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Phone
    m = _PHONE_RE.search(text)
    if m:
        out["phone"] = m.group(1).strip()

    # Email
    m = _EMAIL_RE.search(text)
    if m:
        out["email"] = m.group(1).strip()

    # Incident location
    m = _LOCATION_RE.search(text)
    if m:
        out["incident_location"] = m.group(1).strip()

    return out


# ---------------------------------------------------------------------------
# LLM-assisted extraction
# ---------------------------------------------------------------------------
_LLM_SYSTEM_PROMPT = """You are a claim-form field extractor. Given OCR'd text from an insurance claim form, extract the requested field. Return ONLY the value as a JSON object with a single key matching the requested field name. If the field is not present in the text, return {"<field>": null}.

Examples:
- Requested: policyholder_name → {"policyholder_name": "Alice Homeowner"}
- Requested: date_of_loss → {"date_of_loss": "2026-03-14"}
- Requested: damage_description → {"damage_description": "Wind damage to roof shingles during storm."}
"""


def _llm_extract_field(
    field_name: str, *, ocr_text: str, config: IntakeConfig,
) -> tuple[Any, bool]:
    """Ask the LLM to extract one field. Returns (value, success)."""
    if not config.llm_enabled:
        return None, False

    try:
        from openai import OpenAI
    except ImportError:
        logger.debug("openai SDK not installed; skipping LLM extraction")
        return None, False

    user_prompt = (
        f"Extract the field '{field_name}' from the following OCR text. "
        f"Return JSON with a single key '{field_name}'.\n\n"
        f"OCR text:\n```\n{ocr_text[:4000]}\n```"
    )

    try:
        client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            timeout=config.llm_timeout_sec,
        )
        resp = client.chat.completions.create(
            model=config.llm_model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        # Strip markdown fences if present.
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
        value = parsed.get(field_name)
        if value is None:
            return None, False
        # Special-case claim_type and date for normalisation.
        if field_name == "claim_type":
            return _normalise_claim_type(str(value)).value, True
        if field_name == "date_of_loss":
            return _normalise_date(str(value)), True
        return str(value).strip(), True
    except Exception as exc:
        logger.debug("LLM extraction failed for %s: %s", field_name, exc)
        return None, False


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
#: The fields the extractor tries to populate, in priority order.
_TARGET_FIELDS: tuple[str, ...] = (
    "policyholder_name",
    "policy_id",
    "claim_type",
    "date_of_loss",
    "damage_description",
    "amount_claimed",
    "incident_location",
    "phone",
    "email",
)


def extract(
    ocr_text: str, *,
    config: IntakeConfig | None = None,
    passthrough: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Extract structured fields from OCR text.

    ``passthrough`` is a dict of fields already provided by the caller
    (e.g. the web portal submitted ``policy_id`` directly). These are
    trusted as-is and not re-extracted.
    """
    cfg = config or IntakeConfig.from_env()
    passthrough = passthrough or {}

    result = ExtractionResult()
    llm_used = False

    # 1. Seed with passthrough values
    for k, v in passthrough.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        result.fields[k] = v
        result.method[k] = "passthrough"

    # 2. Regex pass on whatever's missing
    regex_hits = _regex_extract(ocr_text) if ocr_text else {}
    for field_name in _TARGET_FIELDS:
        if field_name in result.fields:
            continue
        val = regex_hits.get(field_name)
        if val is not None and val != "":
            # Normalise claim_type / date
            if field_name == "claim_type" and isinstance(val, str):
                val = _normalise_claim_type(val).value
            elif field_name == "date_of_loss" and isinstance(val, str):
                val = _normalise_date(val)
            result.fields[field_name] = val
            result.method[field_name] = "regex"

    # 3. LLM pass on whatever's still missing
    for field_name in _TARGET_FIELDS:
        if field_name in result.fields:
            continue
        value, ok = _llm_extract_field(field_name, ocr_text=ocr_text, config=cfg)
        if ok and value:
            result.fields[field_name] = value
            result.method[field_name] = "llm"
            llm_used = True

    # 4. Tally missing fields
    for field_name in _TARGET_FIELDS:
        if field_name not in result.fields or not result.fields[field_name]:
            result.method[field_name] = "missing"
            result.missing_count += 1

    result.llm_used = llm_used
    return result
