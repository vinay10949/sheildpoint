"""
Field validation for the claim intake pipeline (SP-203).

The validator takes the raw extracted fields (a flat dict) and produces a
:class:`ValidationResult` containing:

- The cleaned field values, ready to feed into :class:`StandardClaim`.
- A list of :class:`FieldError` objects, one per invalid/missing field.
- A boolean ``ok`` flag — True iff the claim can be promoted to accepted.

Validation rules
----------------

Required fields (per SP-203 AC):
- ``policyholder_name`` — must be non-empty, contain at least 2 alphabetic
  characters, and not look like OCR garbage (e.g. all-caps block of >50
  chars with no spaces).
- ``policy_id`` — must match the canonical format ``[A-Z]{2,4}-\\d{4}-\\d{3,5}``.
- ``claim_type`` — must round-trip through the :class:`ClaimType` enum.
- ``date_of_loss`` — must be a valid ISO-8601 date and not in the future.
- ``damage_description`` — must be at least 10 characters (single-word
  "Wind" is too vague to route a claim).

Optional fields (validated if present, but their absence is not an error):
- ``amount_claimed`` — must be a non-negative number.
- ``phone`` — must be a plausible phone number (10+ digits).
- ``email`` — must be a plausible email.

If ``len(errors) > config.review_max_missing_fields`` (default 5), the
claim is rejected outright instead of going to review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .config import IntakeConfig
from .schemas import (
    ClaimType,
    FieldError,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    StandardClaim,
)

logger = logging.getLogger("claim_intake.validator")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """The outcome of validating a set of extracted fields."""

    #: Cleaned values that passed validation (subset of input).
    cleaned: dict[str, Any] = field(default_factory=dict)
    #: All errors encountered, with field-level detail.
    errors: list[FieldError] = field(default_factory=list)
    #: Names of required fields that were missing entirely.
    missing: list[str] = field(default_factory=list)
    #: Names of optional fields that were present but invalid.
    invalid_optional: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff the claim can be promoted to accepted status."""
        return not self.errors

    @property
    def missing_required_count(self) -> int:
        return len(self.missing)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
_POLICY_ID_RE = re.compile(r"^[A-Z]{2,4}-\d{4}-\d{3,5}$")
_PHONE_DIGITS_RE = re.compile(r"\d")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
#: A name must contain at least two alphabetic characters and at least one
#: space (single-token "CLAIMANT" is rejected as OCR garbage).
_NAME_RE = re.compile(r"^[A-Z][A-Za-z'\-\.]+(?:\s+[A-Z][A-Za-z'\-\.]+)+$")


# ---------------------------------------------------------------------------
# Field-level validators
# ---------------------------------------------------------------------------
def _validate_policyholder_name(value: Any) -> tuple[str | None, FieldError | None]:
    if not isinstance(value, str) or not value.strip():
        return None, FieldError(
            field="policyholder_name", message="Policyholder name is required."
        )
    v = value.strip()
    # Title-case the name (OCR often returns all-caps or all-lower).
    if v.isupper() or v.islower():
        v = v.title()
    if not _NAME_RE.match(v):
        return None, FieldError(
            field="policyholder_name",
            message=(
                f"'{v}' does not look like a full name "
                "(expected at least first and last name)."
            ),
        )
    return v, None


def _validate_policy_id(value: Any) -> tuple[str | None, FieldError | None]:
    if not isinstance(value, str) or not value.strip():
        return None, FieldError(
            field="policy_id", message="Policy ID is required."
        )
    v = value.strip().upper()
    if not _POLICY_ID_RE.match(v):
        return None, FieldError(
            field="policy_id",
            message=(
                f"'{value}' is not a valid policy ID "
                "(expected format: XX-YYYY-NNN)."
            ),
        )
    return v, None


def _validate_claim_type(value: Any) -> tuple[ClaimType | None, FieldError | None]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, FieldError(
            field="claim_type", message="Claim type is required."
        )
    if isinstance(value, ClaimType):
        return value, None
    v = str(value).strip().lower()
    try:
        return ClaimType(v), None
    except ValueError:
        return None, FieldError(
            field="claim_type",
            message=(
                f"'{value}' is not a known claim type "
                f"(valid: {[ct.value for ct in ClaimType]})."
            ),
        )


def _validate_date_of_loss(value: Any) -> tuple[str | None, FieldError | None]:
    if not isinstance(value, str) or not value.strip():
        return None, FieldError(
            field="date_of_loss", message="Date of loss is required."
        )
    v = value.strip()
    try:
        parsed = datetime.strptime(v, "%Y-%m-%d").date()
    except ValueError:
        return None, FieldError(
            field="date_of_loss",
            message=f"'{value}' is not a valid ISO date (YYYY-MM-DD).",
        )
    today = date.today()
    if parsed > today:
        return None, FieldError(
            field="date_of_loss",
            message=f"Date of loss '{v}' is in the future (today: {today.isoformat()}).",
        )
    # Reject dates more than 5 years in the past — likely OCR error or fraud.
    five_years_ago = date(today.year - 5, today.month, today.day)
    if parsed < five_years_ago:
        return None, FieldError(
            field="date_of_loss",
            message=(
                f"Date of loss '{v}' is more than 5 years ago — "
                "verify the OCR output."
            ),
        )
    return v, None


def _validate_damage_description(value: Any) -> tuple[str | None, FieldError | None]:
    if not isinstance(value, str) or not value.strip():
        return None, FieldError(
            field="damage_description", message="Damage description is required."
        )
    v = value.strip()
    if len(v) < 10:
        return None, FieldError(
            field="damage_description",
            message=(
                f"Damage description too short ({len(v)} chars; "
                "minimum 10)."
            ),
        )
    # Reject obvious OCR garbage: a single line of >200 chars with no spaces.
    if "\n" not in v and len(v) > 200 and " " not in v:
        return None, FieldError(
            field="damage_description",
            message="Damage description looks like OCR garbage (no spaces).",
        )
    return v, None


# ---- Optional fields ------------------------------------------------------
def _validate_amount(value: Any) -> tuple[float | None, FieldError | None]:
    if value is None:
        return None, None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, FieldError(
            field="amount_claimed",
            message=f"'{value}' is not a valid amount.",
        )
    if v < 0:
        return None, FieldError(
            field="amount_claimed",
            message=f"Amount cannot be negative (got {v}).",
        )
    return v, None


def _validate_phone(value: Any) -> tuple[str | None, FieldError | None]:
    if value is None or value == "":
        return None, None
    v = str(value).strip()
    digits = _PHONE_DIGITS_RE.findall(v)
    if len(digits) < 10:
        return None, FieldError(
            field="phone",
            message=f"'{value}' is not a valid phone number (need >=10 digits).",
        )
    return v, None


def _validate_email(value: Any) -> tuple[str | None, FieldError | None]:
    if value is None or value == "":
        return None, None
    v = str(value).strip()
    if not _EMAIL_RE.match(v):
        return None, FieldError(
            field="email", message=f"'{value}' is not a valid email address."
        )
    return v, None


def _validate_incident_location(value: Any) -> tuple[str | None, FieldError | None]:
    if value is None or value == "":
        return None, None
    v = str(value).strip()
    if len(v) < 5:
        return None, FieldError(
            field="incident_location",
            message="Incident location is too short (need >=5 chars).",
        )
    return v, None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
#: Maps field name → validator function. Returns (cleaned_value, error_or_None).
_REQUIRED_VALIDATORS = {
    "policyholder_name": _validate_policyholder_name,
    "policy_id": _validate_policy_id,
    "claim_type": _validate_claim_type,
    "date_of_loss": _validate_date_of_loss,
    "damage_description": _validate_damage_description,
}

_OPTIONAL_VALIDATORS = {
    "amount_claimed": _validate_amount,
    "phone": _validate_phone,
    "email": _validate_email,
    "incident_location": _validate_incident_location,
}


def validate(
    fields: dict[str, Any], *, config: IntakeConfig | None = None,
) -> ValidationResult:
    """Validate a flat dict of extracted fields.

    Returns a :class:`ValidationResult` with cleaned values and errors.
    Does NOT raise — every problem becomes a :class:`FieldError`.
    """
    cfg = config or IntakeConfig.from_env()
    result = ValidationResult()

    # Required fields
    for name in REQUIRED_FIELDS:
        validator = _REQUIRED_VALIDATORS[name]
        value = fields.get(name)
        cleaned, err = validator(value)
        if err is not None:
            result.errors.append(err)
            if value is None or (isinstance(value, str) and not value.strip()):
                result.missing.append(name)
        elif cleaned is not None:
            result.cleaned[name] = cleaned

    # Optional fields
    for name in OPTIONAL_FIELDS:
        validator = _OPTIONAL_VALIDATORS.get(name)
        if validator is None:
            continue
        value = fields.get(name)
        cleaned, err = validator(value)
        if err is not None:
            result.errors.append(err)
            result.invalid_optional.append(name)
        elif cleaned is not None:
            result.cleaned[name] = cleaned

    return result


def to_standard_claim(cleaned: dict[str, Any]) -> StandardClaim:
    """Build a :class:`StandardClaim` from a cleaned field dict.

    Raises ``pydantic.ValidationError`` if any required field is missing
    or malformed — but in practice :func:`validate` has already filtered
    those out, so this should only fail on a programming error.
    """
    return StandardClaim(**{
        k: v for k, v in cleaned.items()
        if k in StandardClaim.model_fields
    })
