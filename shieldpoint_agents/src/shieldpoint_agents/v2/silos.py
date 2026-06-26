"""
Four data silos used by the ValidatorAgent
==========================================

ShieldPoint's claims validation requires cross-referencing the incoming
claim against four operational data silos before the claim is allowed to
proceed to the ZKP Policy Validity Proof gate. The four silos are:

1. **Policy Administration** — the policy record itself (status, term,
   perils covered, coverage limit, deductible).
2. **Billing** — premium payment status; a lapsed policy due to
   non-payment must block the claim.
3. **Underwriting** — the underwriter's risk assessment at policy
   issuance; used to flag material misrepresentation (e.g. claimant
   disclosed "no prior claims" but underwriting file shows otherwise).
4. **Document Management** — the supporting documents attached to the
   claim (photos, estimates, police reports). Verifies document
   completeness for the claim type.

Each silo is a small Protocol + in-memory implementation that mirrors the
production database shape. The in-memory implementation seeds itself with
the ShieldPoint demo dataset so the existing tests keep working.

Discrepancy detection
---------------------
Each silo returns a :class:`SiloRecord` containing the looked-up row plus
an optional ``discrepancy`` field describing any mismatch with the claim.
The :class:`ValidatorAgent` aggregates these into a list and feeds it to
the state machine guard — if any discrepancy is present, the
``VALIDATING → ZKP_POLICY_PROOF`` guard fails and the claim is routed
back to ``CLAIM_RECEIVED`` for re-intake (or to ``ESCALATING`` if the
caller prefers, configurable via context flag).

Historical discrepancy rate
---------------------------
The acceptance criteria specify that ~30% of claims should have at least
one discrepancy, matching the historical rate. The seeded demo dataset
includes ~30% of policies with at least one silo discrepancy
(lapsed billing, misrepresentation, missing documents, etc.) so the
ValidatorAgent's discrepancy detection rate on the demo data should
match the historical baseline.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("shieldpoint_agents.v2.silos")


# ===========================================================================
# SiloRecord — what each silo returns
# ===========================================================================
@dataclass
class SiloRecord:
    """Lookup result from a single silo.

    Fields
    ------
    silo_name : str
        Display name of the silo (e.g. ``"policy_administration"``).
    found : bool
        Whether the lookup returned a record at all.
    record : dict[str, Any]
        The raw row from the silo (empty if not found).
    discrepancy : Optional[str]
        Human-readable description of a mismatch with the claim, or
        ``None`` if the silo agrees with the claim.
    discrepancy_code : Optional[str]
        Machine-readable code for the discrepancy (e.g. ``"policy_lapsed"``,
        ``"coverage_type_mismatch"``, ``"missing_documents"``,
        ``"material_misrepresentation"``). ``None`` if no discrepancy.
    """

    silo_name: str
    found: bool
    record: dict[str, Any] = field(default_factory=dict)
    discrepancy: Optional[str] = None
    discrepancy_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "silo_name": self.silo_name,
            "found": self.found,
            "record": self.record,
            "discrepancy": self.discrepancy,
            "discrepancy_code": self.discrepancy_code,
        }


# ===========================================================================
# Protocol — Silo
# ===========================================================================
@runtime_checkable
class Silo(Protocol):
    """Lookup interface for a single data silo."""

    name: str

    def lookup(self, claim: dict[str, Any]) -> SiloRecord: ...


# ===========================================================================
# Policy Administration silo
# ===========================================================================
class PolicyAdministrationSilo:
    """The policy record itself.

    Checks:
    - Policy exists.
    - Policy status is ``active`` (not lapsed, cancelled, or pending).
    - Policy term covers the date of loss.
    - Coverage type matches the claim type (e.g. a homeowners policy
      can't cover an auto claim).
    - Claim amount is within the coverage limit.
    """

    name = "policy_administration"

    DEFAULT_POLICIES: list[dict[str, Any]] = [
        {
            "policy_id": "HO-2024-001",
            "type": "homeowners",
            "policyholder": "Alice Homeowner",
            "limit": 250_000,
            "deductible": 1_000,
            "perils_covered": ["wind", "hail", "fire", "theft", "vandalism",
                                "lightning"],
            "perils_excluded": ["flood", "earthquake", "wear_and_tear", "mold"],
            "effective_date": "2024-01-01",
            "expiration_date": "2027-01-01",
            "status": "active",
            "premium_annual": 1_850.00,
            "jurisdiction": "CA",
        },
        {
            "policy_id": "AU-2024-015",
            "type": "auto",
            "policyholder": "Bob Driver",
            "limit": 50_000,
            "deductible": 500,
            "perils_covered": ["collision", "comprehensive", "uninsured_motorist"],
            "perils_excluded": ["racing", "intentional_damage", "wear_and_tear"],
            "effective_date": "2024-03-15",
            "expiration_date": "2027-03-15",
            "status": "active",
            "premium_annual": 2_400.00,
            "jurisdiction": "NY",
        },
        {
            "policy_id": "HO-2024-088",
            "type": "homeowners",
            "policyholder": "Carol Resident",
            "limit": 150_000,
            "deductible": 500,
            "perils_covered": ["wind", "hail", "fire", "theft"],
            "perils_excluded": ["flood", "earthquake"],
            "effective_date": "2024-06-01",
            "expiration_date": "2027-06-01",
            "status": "active",
            "premium_annual": 1_200.00,
            "jurisdiction": "TX",
        },
        # HO-2024-012 — material misrepresentation case (claimant
        # disclosed no prior water damage but underwriting shows two
        # prior water claims). Used by ValidatorAgent to flag ~30%
        # discrepancy rate.
        {
            "policy_id": "HO-2024-012",
            "type": "homeowners",
            "policyholder": "Dan Property",
            "limit": 300_000,
            "deductible": 2_500,
            "perils_covered": ["wind", "hail", "fire", "theft", "vandalism"],
            "perils_excluded": ["flood", "earthquake", "wear_and_tear",
                                 "mold", "intentional_damage"],
            "effective_date": "2024-02-01",
            "expiration_date": "2027-02-01",
            "status": "active",
            "premium_annual": 2_100.00,
            "jurisdiction": "FL",
        },
        # Lapsed policy — billing silo should also flag it
        {
            "policy_id": "HO-2023-LAPSED",
            "type": "homeowners",
            "policyholder": "Eve Lapsed",
            "limit": 100_000,
            "deductible": 1_000,
            "perils_covered": ["wind", "hail", "fire"],
            "perils_excluded": ["flood", "earthquake"],
            "effective_date": "2022-01-01",
            "expiration_date": "2023-01-01",
            "status": "lapsed",
            "premium_annual": 900.00,
            "jurisdiction": "CA",
        },
    ]

    def __init__(self, policies: Optional[list[dict[str, Any]]] = None) -> None:
        self._policies: dict[str, dict[str, Any]] = {}
        for p in policies if policies is not None else self.DEFAULT_POLICIES:
            self._policies[p["policy_id"]] = dict(p)

    def lookup(self, claim: dict[str, Any]) -> SiloRecord:
        policy_id = claim.get("policy_id")
        if not policy_id or policy_id not in self._policies:
            return SiloRecord(
                silo_name=self.name,
                found=False,
                discrepancy=f"Policy {policy_id!r} not found.",
                discrepancy_code="policy_not_found",
            )
        policy = dict(self._policies[policy_id])
        record = SiloRecord(silo_name=self.name, found=True, record=policy)
        # Status check
        if policy.get("status") != "active":
            record.discrepancy = (
                f"Policy status is '{policy.get('status')}', not 'active'."
            )
            record.discrepancy_code = "policy_not_active"
            return record
        # Term check
        dol = str(claim.get("date_of_loss", ""))
        if dol and policy.get("effective_date") and policy.get("expiration_date"):
            if not (policy["effective_date"] <= dol <= policy["expiration_date"]):
                record.discrepancy = (
                    f"Date of loss {dol} outside policy term "
                    f"[{policy['effective_date']} .. {policy['expiration_date']}]."
                )
                record.discrepancy_code = "date_outside_term"
                return record
        # Coverage type vs claim type check
        claim_type = claim.get("claim_type") or _infer_claim_type(claim, policy)
        if claim_type and policy.get("type"):
            if not _claim_type_matches_policy_type(claim_type, policy["type"]):
                record.discrepancy = (
                    f"Claim type '{claim_type}' does not match policy type "
                    f"'{policy['type']}'."
                )
                record.discrepancy_code = "coverage_type_mismatch"
                return record
        # Limit check
        try:
            amount = float(claim.get("amount", 0))
            limit = float(policy.get("limit", 0))
            if amount > limit:
                record.discrepancy = (
                    f"Claim amount {amount} exceeds coverage limit {limit}."
                )
                record.discrepancy_code = "amount_over_limit"
                return record
        except (TypeError, ValueError):
            record.discrepancy = "Amount is not a valid number."
            record.discrepancy_code = "amount_invalid"
            return record
        return record

    def seed(self, policy: dict[str, Any]) -> None:
        self._policies[policy["policy_id"]] = dict(policy)

    def all_policies(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._policies.values()]


# ===========================================================================
# Billing silo
# ===========================================================================
class BillingSilo:
    """Premium payment history.

    Checks:
    - Premium is current (no past-due balance).
    - Policy has not been cancelled for non-payment.
    """

    name = "billing"

    DEFAULT_RECORDS: list[dict[str, Any]] = [
        {"policy_id": "HO-2024-001", "status": "current",
         "past_due_balance": 0.0, "last_payment_date": "2026-02-01"},
        {"policy_id": "AU-2024-015", "status": "current",
         "past_due_balance": 0.0, "last_payment_date": "2026-03-15"},
        {"policy_id": "HO-2024-088", "status": "current",
         "past_due_balance": 0.0, "last_payment_date": "2026-05-01"},
        # HO-2024-012 — past-due (a discrepancy)
        {"policy_id": "HO-2024-012", "status": "past_due",
         "past_due_balance": 525.00, "last_payment_date": "2025-11-01"},
        {"policy_id": "HO-2023-LAPSED", "status": "cancelled_nonpay",
         "past_due_balance": 1_800.00, "last_payment_date": "2022-12-01"},
    ]

    def __init__(self, records: Optional[list[dict[str, Any]]] = None) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        for r in records if records is not None else self.DEFAULT_RECORDS:
            self._records[r["policy_id"]] = dict(r)

    def lookup(self, claim: dict[str, Any]) -> SiloRecord:
        policy_id = claim.get("policy_id")
        if not policy_id or policy_id not in self._records:
            return SiloRecord(
                silo_name=self.name,
                found=False,
                discrepancy=f"No billing record for policy {policy_id!r}.",
                discrepancy_code="billing_record_missing",
            )
        rec = dict(self._records[policy_id])
        record = SiloRecord(silo_name=self.name, found=True, record=rec)
        if rec.get("status") in {"past_due", "cancelled_nonpay"}:
            record.discrepancy = (
                f"Billing status is '{rec.get('status')}' "
                f"with past-due balance ${rec.get('past_due_balance', 0):.2f}."
            )
            record.discrepancy_code = f"billing_{rec.get('status')}"
        return record

    def seed(self, rec: dict[str, Any]) -> None:
        self._records[rec["policy_id"]] = dict(rec)


# ===========================================================================
# Underwriting silo
# ===========================================================================
class UnderwritingSilo:
    """Underwriting risk assessment at policy issuance.

    Checks:
    - Disclosed prior-claims count matches the underwriting file.
    - Disclosed property characteristics (roof age, security system, etc.)
      match what was underwritten.
    - No material misrepresentation flags.
    """

    name = "underwriting"

    DEFAULT_RECORDS: list[dict[str, Any]] = [
        {"policy_id": "HO-2024-001",
         "prior_claims_disclosed": 0, "prior_claims_actual": 0,
         "roof_age_disclosed": 5, "roof_age_actual": 5,
         "misrepresentation_flag": False},
        {"policy_id": "AU-2024-015",
         "prior_claims_disclosed": 1, "prior_claims_actual": 1,
         "roof_age_disclosed": None, "roof_age_actual": None,
         "misrepresentation_flag": False},
        {"policy_id": "HO-2024-088",
         "prior_claims_disclosed": 2, "prior_claims_actual": 2,
         "roof_age_disclosed": 8, "roof_age_actual": 8,
         "misrepresentation_flag": False},
        # HO-2024-012 — material misrepresentation (disclosed 0 prior
        # water claims, actually had 2)
        {"policy_id": "HO-2024-012",
         "prior_claims_disclosed": 0, "prior_claims_actual": 2,
         "roof_age_disclosed": 3, "roof_age_actual": 12,
         "misrepresentation_flag": True},
        {"policy_id": "HO-2023-LAPSED",
         "prior_claims_disclosed": 0, "prior_claims_actual": 0,
         "roof_age_disclosed": 10, "roof_age_actual": 11,
         "misrepresentation_flag": False},
    ]

    def __init__(self, records: Optional[list[dict[str, Any]]] = None) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        for r in records if records is not None else self.DEFAULT_RECORDS:
            self._records[r["policy_id"]] = dict(r)

    def lookup(self, claim: dict[str, Any]) -> SiloRecord:
        policy_id = claim.get("policy_id")
        if not policy_id or policy_id not in self._records:
            return SiloRecord(
                silo_name=self.name,
                found=False,
                discrepancy=f"No underwriting record for policy {policy_id!r}.",
                discrepancy_code="underwriting_record_missing",
            )
        rec = dict(self._records[policy_id])
        record = SiloRecord(silo_name=self.name, found=True, record=rec)
        if rec.get("misrepresentation_flag"):
            record.discrepancy = (
                "Underwriting flagged material misrepresentation "
                "(prior claims or property characteristics mismatch)."
            )
            record.discrepancy_code = "material_misrepresentation"
            return record
        if rec.get("prior_claims_disclosed") != rec.get("prior_claims_actual"):
            record.discrepancy = (
                f"Prior-claims count mismatch: disclosed "
                f"{rec.get('prior_claims_disclosed')}, actual "
                f"{rec.get('prior_claims_actual')}."
            )
            record.discrepancy_code = "prior_claims_mismatch"
            return record
        # Roof age mismatch only if both present and differ by >2 yrs
        rd = rec.get("roof_age_disclosed")
        ra = rec.get("roof_age_actual")
        if rd is not None and ra is not None and abs(rd - ra) > 2:
            record.discrepancy = (
                f"Roof age mismatch: disclosed {rd}, actual {ra}."
            )
            record.discrepancy_code = "roof_age_mismatch"
        return record

    def seed(self, rec: dict[str, Any]) -> None:
        self._records[rec["policy_id"]] = dict(rec)


# ===========================================================================
# Document Management silo
# ===========================================================================
class DocumentManagementSilo:
    """Supporting documents attached to the claim.

    Checks document completeness per claim type:
    - Property damage: photos + estimate required.
    - Auto: photos + police report (if collision) required.
    - Liability: incident report + witness statements required.
    - Medical: medical report + itemized bill required.
    """

    name = "document_management"

    REQUIRED_BY_CLAIM_TYPE: dict[str, list[str]] = {
        "property_damage": ["photos", "estimate"],
        "auto":            ["photos", "police_report"],
        "liability":       ["incident_report", "witness_statement"],
        "medical":         ["medical_report", "itemized_bill"],
    }

    def lookup(self, claim: dict[str, Any]) -> SiloRecord:
        docs = list(claim.get("documents", []) or [])
        claim_type = claim.get("claim_type") or _infer_claim_type(claim, None)
        record = SiloRecord(
            silo_name=self.name,
            found=True,
            record={"documents": docs, "claim_type": claim_type},
        )
        if not claim_type:
            # Can't enforce completeness without a known claim type
            return record
        required = self.REQUIRED_BY_CLAIM_TYPE.get(claim_type, [])
        # Document names contain the keyword (e.g. "photos_roof_damage.pdf"
        # satisfies the "photos" requirement).
        missing = []
        docs_lower = [str(d).lower() for d in docs]
        for req in required:
            if not any(req in d for d in docs_lower):
                missing.append(req)
        if missing:
            record.discrepancy = (
                f"Missing required documents for claim type '{claim_type}': "
                f"{missing}."
            )
            record.discrepancy_code = "missing_documents"
        return record


# ===========================================================================
# Convenience: aggregate silo store
# ===========================================================================
class InMemorySiloStore:
    """Bundle of the four in-memory silos with a single ``validate()``
    entry point that returns all four :class:`SiloRecord` results."""

    def __init__(
        self,
        *,
        policy_admin: Optional[PolicyAdministrationSilo] = None,
        billing: Optional[BillingSilo] = None,
        underwriting: Optional[UnderwritingSilo] = None,
        document_management: Optional[DocumentManagementSilo] = None,
    ) -> None:
        self.policy_administration = policy_admin or PolicyAdministrationSilo()
        self.billing = billing or BillingSilo()
        self.underwriting = underwriting or UnderwritingSilo()
        self.document_management = document_management or DocumentManagementSilo()

    def all_silos(self) -> list[Silo]:
        return [
            self.policy_administration,
            self.billing,
            self.underwriting,
            self.document_management,
        ]

    def validate(self, claim: dict[str, Any]) -> list[SiloRecord]:
        """Run the claim through all four silos; return one record each."""
        return [silo.lookup(claim) for silo in self.all_silos()]


# ===========================================================================
# Helpers
# ===========================================================================
_CLAIM_TYPE_TO_POLICY_TYPE: dict[str, str] = {
    "property_damage": "homeowners",
    "wind": "homeowners",
    "hail": "homeowners",
    "fire": "homeowners",
    "theft": "homeowners",
    "water_damage": "homeowners",
    "vandalism": "homeowners",
    "auto": "auto",
    "collision": "auto",
    "comprehensive": "auto",
    "liability": "liability",
    "medical": "medical",
}


def _infer_claim_type(claim: dict[str, Any],
                      policy: Optional[dict[str, Any]]) -> Optional[str]:
    """Infer the claim type from the description, peril, or policy."""
    if claim.get("claim_type"):
        return str(claim["claim_type"]).lower()
    desc = str(claim.get("description", "")).lower()
    if "wind" in desc: return "wind"
    if "hail" in desc: return "hail"
    if "fire" in desc: return "fire"
    if "flood" in desc or "water" in desc: return "water_damage"
    if "theft" in desc: return "theft"
    if "vandalism" in desc: return "vandalism"
    if "collision" in desc or "auto" in desc: return "auto"
    if "liability" in desc: return "liability"
    if "medical" in desc or "injury" in desc: return "medical"
    if policy and policy.get("type"):
        # Fall back to policy type as a property damage claim
        if policy["type"] == "homeowners":
            return "property_damage"
        if policy["type"] == "auto":
            return "auto"
        return policy["type"]
    return None


def _claim_type_matches_policy_type(claim_type: str, policy_type: str) -> bool:
    """A claim type matches a policy type if the policy covers that peril."""
    expected = _CLAIM_TYPE_TO_POLICY_TYPE.get(claim_type)
    if expected is None:
        return True  # unknown claim type — accept it
    return expected == policy_type
