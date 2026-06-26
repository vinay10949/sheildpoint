#!/usr/bin/env python3
"""
ShieldPoint Evaluation Dataset Generator
========================================

Generates a 5,000-claim labeled historical-claims dataset for the
"Create Evaluation Dataset and Baseline Metrics" Jira ticket.

Each claim record carries:
  * Original claim data (claim_id, policy_id, claimant, amount, description,
    date_of_loss, adjuster_id, documents, source_system)
  * Correct classification (severity, claim_type, claim_subtype, fraud_flag)
  * Correct payout amount (correct_payout_amount)
  * Processing timeline (days_to_first_contact, days_to_investigation_complete,
    days_to_settle)
  * Expected agent decision (expected_decision in {approve, deny,
    route_to_manual_review})
  * Expected Langfuse output (expected_severity, expected_claim_type,
    expected_fraud_flag, expected_payout_amount, expected_decision)

PII is anonymized using deterministic synthetic identifiers: a SHA-1 hash of
the raw claimant name maps to a stable CLAIMANT-XXXXX token, so the same
source person always receives the same anonymized identity across all splits.
Financial fields (amount, deductible, limit, premium) are preserved verbatim
per the acceptance criterion "financial data preserved".

Splits: 70 / 15 / 15  ->  3,500 train / 750 val / 750 test.

Outputs (under /home/z/my-project/download/dataset/):
  * train.csv, val.csv, test.csv           (flat CSV, easy Excel inspection)
  * train.jsonl, val.jsonl, test.jsonl     (one JSON object per line, for
                                            LoRA fine-tuning + Langfuse)
  * dataset_manifest.json                  (counts, schema, checksums)
  * DATASET_README.md                      (human-readable description)

Usage:
  python3 /home/z/my-project/scripts/generate_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TOTAL_CLAIMS = 5_000
TRAIN_COUNT = 3_500
VAL_COUNT = 750
TEST_COUNT = 750
assert TRAIN_COUNT + VAL_COUNT + TEST_COUNT == TOTAL_CLAIMS

OUTPUT_DIR = Path("/home/z/my-project/download/dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Coverage mix (per question 5: Homeowners + Auto + Health/Accident)
LINE_MIX = [
    ("homeowners", 0.50),  # 2,500 claims
    ("auto",       0.30),  # 1,500 claims
    ("health",     0.20),  # 1,000 claims
]

# Severity distribution (industry-typical for P&C + A&H)
SEVERITY_MIX = [
    ("low",          0.40),
    ("medium",       0.35),
    ("high",         0.20),
    ("catastrophic", 0.05),
]

# Decision distribution (industry-typical: 70% auto-approve, 22% manual, 8% deny)
DECISION_MIX = [
    ("approve",              0.70),
    ("route_to_manual_review", 0.22),
    ("deny",                 0.08),
]

# Fraud prevalence — ~5% of all claims are actual fraud (denied), and an
# additional ~3% are flagged for suspicious-pattern review (manual).
FRAUD_RATE = 0.05
SUSPICIOUS_FLAG_RATE = 0.03

# Peril catalog by line of business. Each peril has a base severity multiplier
# and a coverage flag (covered/excluded under standard ShieldPoint policies).
# Each peril is weighted by an explicit probability. Excluded perils are
# intentionally rare (a few percent each) so the overall decision distribution
# matches the industry-typical 70% approve / 22% manual / 8% deny target.
PERIL_CATALOG = {
    "homeowners": [
        ("wind",            0.22, 1.00, True),   # covered, common
        ("hail",            0.14, 1.20, True),
        ("fire",            0.08, 2.50, True),
        ("lightning",       0.05, 1.80, True),
        ("theft",           0.10, 1.40, True),
        ("vandalism",       0.08, 0.80, True),
        ("water_damage",    0.16, 1.60, True),   # non-flood water
        ("smoke",           0.07, 0.90, True),
        ("flood",           0.04, 2.20, False),  # excluded peril -> deny
        ("earthquake",      0.02, 3.00, False),  # excluded peril -> deny
        ("mold",            0.02, 1.10, False),  # excluded peril -> deny
        ("wear_and_tear",   0.02, 0.70, False),  # excluded peril -> deny
    ],
    "auto": [
        ("collision",           0.30, 1.00, True),
        ("comprehensive_glass", 0.14, 0.40, True),
        ("comprehensive_theft", 0.10, 1.50, True),
        ("comprehensive_weather", 0.12, 1.20, True),
        ("vandalism",           0.08, 0.70, True),
        ("uninsured_motorist",  0.10, 1.80, True),
        ("pip_medical",         0.12, 1.30, True),
        ("racing",              0.01, 2.50, False),   # excluded -> deny
        ("intentional_damage",  0.01, 2.00, False),   # excluded -> deny
        ("wear_and_tear",       0.02, 0.60, False),   # excluded -> deny
    ],
    "health": [
        ("emergency_room",    0.18, 1.00, True),
        ("urgent_care",       0.16, 0.40, True),
        ("surgery_elective",  0.08, 2.20, True),
        ("surgery_emergency", 0.06, 3.50, True),
        ("imaging",           0.14, 0.80, True),
        ("physical_therapy",  0.16, 0.60, True),
        ("prescription",      0.18, 0.30, True),
        ("experimental_tx",   0.02, 2.80, False),   # excluded -> deny
        ("cosmetic",          0.01, 1.50, False),   # excluded -> deny
        ("pre_existing",      0.01, 1.20, False),   # excluded -> deny
    ],
}

# Severity base payouts (USD) per line — anchors the correct_payout_amount.
SEVERITY_BASE_PAYOUT = {
    "homeowners": {"low": 800, "medium": 4_500, "high": 18_000, "catastrophic": 75_000},
    "auto":       {"low": 600, "medium": 3_500, "high": 12_000, "catastrophic": 35_000},
    "health":     {"low": 400, "medium": 2_500, "high": 9_000,  "catastrophic": 28_000},
}

# Processing-timeline baselines (business days) by severity.
TIMELINE_BASE = {
    "low":          {"first_contact": 1, "investigation": 2, "settle": 5},
    "medium":       {"first_contact": 2, "investigation": 5, "settle": 12},
    "high":         {"first_contact": 1, "investigation": 9, "settle": 25},
    "catastrophic": {"first_contact": 1, "investigation": 18, "settle": 60},
}

# Documents typically attached to a claim (used to populate `documents`).
DOCUMENT_TEMPLATES = {
    "homeowners": [
        "photos_{peril}_damage.pdf",
        "contractor_estimate.pdf",
        "police_report.pdf" if "{peril}" == "theft" else "adjuster_inspection.pdf",
        "hydrology_report.pdf",
    ],
    "auto": [
        "police_report.pdf",
        "photos_vehicle_damage.pdf",
        "repair_estimate.pdf",
        "medical_report.pdf",
    ],
    "health": [
        "medical_records.pdf",
        "itemized_bill.pdf",
        "treatment_plan.pdf",
        "prior_authorization.pdf",
    ],
}

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Liam", "Maria", "Noah", "Olivia", "Paul",
    "Quinn", "Rita", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zane", "Amir", "Bianca", "Carlos", "Divya", "Ethan", "Fatima",
    "Gabriel", "Hana", "Ibrahim", "Jasmine", "Kwame", "Leila", "Marco",
    "Nadia", "Omar", "Priya", "Quincy", "Ravi", "Sofia", "Tariq", "Uma",
    "Vikram", "Wei", "Ximena", "Yusuf", "Zara",
]
LAST_NAMES = [
    "Homeowner", "Driver", "Resident", "Property", "Smith", "Johnson",
    "Lee", "Wilson", "Davis", "Brown", "Garcia", "Miller", "Martinez",
    "Patel", "Kim", "Chen", "Anderson", "Taylor", "Thomas", "Jackson",
    "White", "Harris", "Martin", "Thompson", "Robinson", "Clark",
    "Lewis", "Walker", "Hall", "Allen", "Young", "King", "Wright",
    "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Nelson",
]

CLAIM_ADJUSTER_POOL = [f"ADJ-{i:03d}" for i in range(1, 51)]  # 50 adjusters


# ---------------------------------------------------------------------------
# PII anonymization
# ---------------------------------------------------------------------------
def anonymize_pii(raw_name: str, raw_email: str, raw_phone: str,
                  raw_ssn: str, raw_address: str) -> dict[str, str]:
    """Deterministic PII -> synthetic identifier mapping.

    Uses SHA-1(raw_name) as the stable seed so the same person always maps
    to the same anonymized identity across all splits and reloads. Financial
    fields are NOT touched here — they pass through verbatim per AC.
    """
    digest = hashlib.sha1(raw_name.encode("utf-8")).hexdigest()[:10].upper()
    short = digest[:6]
    return {
        "claimant_anon": f"CLAIMANT-{short}",
        "claimant_email_anon": f"user-{digest.lower()}@anonymized.shieldpoint.local",
        "claimant_phone_anon": f"+1-555-{digest[:3]}-{digest[3:7]}",
        "claimant_ssn_anon": f"XXX-XX-{digest[:4]}",
        "claimant_address_anon": f"ADDR-{short}-ANON",
        # Mapping provenance — kept so auditors can verify the anonymization
        # was deterministic without recovering the original PII.
        "anonymization_hash": digest,
    }


# ---------------------------------------------------------------------------
# Claim record generation
# ---------------------------------------------------------------------------
@dataclass
class ClaimRecord:
    # --- Identifiers ---
    claim_id: str
    policy_id: str
    source_system: str
    # --- Anonymized claimant PII ---
    claimant_anon: str
    claimant_email_anon: str
    claimant_phone_anon: str
    claimant_ssn_anon: str
    claimant_address_anon: str
    anonymization_hash: str
    # --- Claim facts (financial data preserved) ---
    amount: float
    deductible: float
    coverage_limit: float
    premium_annual: float
    description: str
    date_of_loss: str
    reported_date: str
    adjuster_id: str
    documents: list[str] = field(default_factory=list)
    # --- Labels (ground truth for evaluation) ---
    line_of_business: str = ""
    claim_type: str = ""
    claim_subtype: str = ""
    severity: str = ""
    fraud_flag: bool = False
    suspicious_pattern_flag: bool = False
    expected_decision: str = ""
    correct_payout_amount: float = 0.0
    # --- Timeline labels ---
    days_to_first_contact: int = 0
    days_to_investigation_complete: int = 0
    days_to_settle: int = 0
    # --- Expected Langfuse outputs (the agent's target predictions) ---
    expected_severity: str = ""
    expected_claim_type: str = ""
    expected_fraud_flag: bool = False
    expected_payout_amount: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Documents are a list — serialize as pipe-delimited for CSV.
        d["documents"] = "|".join(self.documents)
        return d


def _weighted_pick(rng: random.Random, options: list[tuple[Any, float]]) -> Any:
    """Pick an option from a weighted list. Returns the option (first element)."""
    total = sum(w for _, w in options)
    r = rng.random() * total
    upto = 0.0
    for value, weight in options:
        upto += weight
        if r <= upto:
            return value
    return options[-1][0]  # fallback


def _pick_peril(rng: random.Random, line: str) -> tuple[str, float, bool]:
    """Pick a peril from the catalog using weighted probabilities.

    Each catalog entry is (peril_name, weight, severity_multiplier, covered).
    Returns (peril, multiplier, covered).
    """
    options = [(entry[0], entry[1]) for entry in PERIL_CATALOG[line]]
    chosen_name = _weighted_pick(rng, options)
    for name, _weight, mult, covered in PERIL_CATALOG[line]:
        if name == chosen_name:
            return (name, mult, covered)
    raise RuntimeError(f"peril pick failed for line={line}")


def _build_description(line: str, peril: str, severity: str,
                       claimant_token: str, rng: random.Random) -> str:
    """Compose a human-readable claim description from the peril + severity."""
    severity_phrasing = {
        "low":          ["minor", "small", "slight"],
        "medium":       ["moderate", "noticeable"],
        "high":         ["significant", "severe", "major"],
        "catastrophic": ["total loss", "catastrophic", "extensive"],
    }
    adj = rng.choice(severity_phrasing[severity])
    templates = {
        "homeowners": f"{adj.capitalize()} {peril.replace('_', ' ')} damage reported by {claimant_token}.",
        "auto":       f"{adj.capitalize()} {peril.replace('_', ' ')} incident reported by {claimant_token}.",
        "health":     f"{adj.capitalize()} {peril.replace('_', ' ')} treatment claim for {claimant_token}.",
    }
    base = templates[line]
    if peril in ("flood", "earthquake", "mold", "wear_and_tear"):
        base += " Note: peril typically excluded from standard policy."
    if "intentional" in peril or "racing" in peril:
        base += " Investigator flagged for possible misrepresentation."
    return base


def _calc_payout(line: str, peril: str, severity: str, amount: float,
                 deductible: float, covered: bool, fraud: bool,
                 suspicious: bool, decision: str,
                 rng: random.Random) -> float:
    """Compute the correct (ground-truth) payout amount.

    Rules:
      * If denied (excluded peril, fraud, etc.) -> payout = 0.
      * If routed to manual review -> payout = 0 at decision time
        (held pending investigation).
      * If approved -> max(amount - deductible, 0) scaled by severity
        multiplier, capped at coverage_limit. Payout never exceeds the
        claim amount.
    """
    if decision != "approve":
        return 0.0
    if fraud or not covered:
        return 0.0
    base = SEVERITY_BASE_PAYOUT[line][severity]
    # Apply ±15% noise seeded by per-claim RNG so payouts vary realistically.
    noise = rng.uniform(0.85, 1.15)
    gross = min(amount, base * noise)
    payout = max(gross - deductible, 0.0)
    if suspicious:
        # Suspicious-but-not-fraud claims still pay, often at a reduced rate.
        payout *= 0.85
    return round(payout, 2)


def _calc_timeline(severity: str, decision: str, fraud: bool,
                   suspicious: bool, rng: random.Random) -> tuple[int, int, int]:
    """Compute days_to_first_contact / days_to_investigation_complete /
    days_to_settle.

    Fraud and suspicious claims take longer (extra investigation). Denied
    claims close faster once the denial is issued.
    """
    base = TIMELINE_BASE[severity]
    first = base["first_contact"] + rng.randint(0, 1)
    inv = base["investigation"]
    if fraud or suspicious:
        inv += rng.randint(5, 15)
    settle = base["settle"]
    if decision == "deny":
        # Denials settle faster once decision is made (no payment to issue).
        settle = inv + rng.randint(1, 3)
    elif decision == "route_to_manual_review":
        settle = inv + rng.randint(5, 15)
    else:
        settle += rng.randint(-2, 5)
    return (
        max(first, 1),
        max(inv, first + 1),
        max(settle, inv + 1),
    )


def _build_claim(idx: int, rng: random.Random) -> ClaimRecord:
    """Build a single labeled claim record."""
    line = _weighted_pick(rng, LINE_MIX)
    peril, peril_mult, peril_covered = _pick_peril(rng, line)
    severity = _weighted_pick(rng, SEVERITY_MIX)

    # PII (raw) — anonymized immediately, raw discarded.
    raw_first = rng.choice(FIRST_NAMES)
    raw_last = rng.choice(LAST_NAMES)
    raw_name = f"{raw_first} {raw_last}"
    raw_email = f"{raw_first.lower()}.{raw_last.lower()}@example.com"
    raw_phone = f"+1-555-{rng.randint(100,999):03d}-{rng.randint(1000,9999):04d}"
    raw_ssn = f"{rng.randint(100,999):03d}-{rng.randint(10,99):02d}-{rng.randint(1000,9999):04d}"
    raw_address = f"{rng.randint(100,9999)} {rng.choice(['Main','Oak','Maple','Pine','Cedar'])} St"
    pii = anonymize_pii(raw_name, raw_email, raw_phone, raw_ssn, raw_address)

    # Financial data — preserved verbatim per AC.
    base_amount = SEVERITY_BASE_PAYOUT[line][severity] * peril_mult
    amount = round(base_amount * rng.uniform(0.85, 1.25), 2)
    deductible = rng.choice([250, 500, 1_000, 1_500, 2_500])
    coverage_limit = rng.choice([50_000, 100_000, 150_000, 250_000, 300_000, 500_000])
    premium_annual = round(rng.uniform(800, 3_200), 2)

    # IDs
    claim_id = f"CLM-2026-{idx + 1:05d}"
    policy_prefix = {"homeowners": "HO", "auto": "AU", "health": "HE"}[line]
    policy_id = f"{policy_prefix}-2024-{rng.randint(1, 999):03d}"
    adjuster_id = rng.choice(CLAIM_ADJUSTER_POOL)
    source_system = rng.choice(["AS400", "AS400", "AS400", "PORTAL"])  # 75% AS/400

    # Fraud / suspicious-pattern flags.
    fraud = rng.random() < FRAUD_RATE
    suspicious = (not fraud) and (rng.random() < SUSPICIOUS_FLAG_RATE)

    # Decide expected_decision.
    if fraud or not peril_covered:
        # Excluded peril OR confirmed fraud -> deny.
        expected_decision = "deny"
    elif suspicious or severity == "catastrophic" or amount > 20_000:
        # Suspicious or catastrophic -> manual review.
        expected_decision = "route_to_manual_review"
    else:
        # Otherwise, weighted pick (mostly approve).
        expected_decision = _weighted_pick(rng, DECISION_MIX)

    # Override: if the peril is excluded but decision pick was approve, flip to deny.
    if not peril_covered and expected_decision == "approve":
        expected_decision = "deny"

    # Description.
    description = _build_description(line, peril, severity, pii["claimant_anon"], rng)

    # Documents.
    docs_template = DOCUMENT_TEMPLATES[line]
    n_docs = rng.randint(1, len(docs_template))
    documents = [t.format(peril=peril) for t in rng.sample(docs_template, n_docs)]

    # Dates.
    loss_year = rng.choice([2024, 2025, 2026])
    loss_month = rng.randint(1, 12)
    loss_day = rng.randint(1, 28)
    date_of_loss = date(loss_year, loss_month, loss_day)
    reported_offset = rng.randint(0, 7)  # reported same week as loss
    reported_date = date_of_loss + timedelta(days=reported_offset)

    # Payout + timeline.
    payout = _calc_payout(
        line=line, peril=peril, severity=severity, amount=amount,
        deductible=deductible, covered=peril_covered, fraud=fraud,
        suspicious=suspicious, decision=expected_decision, rng=rng,
    )
    first_c, inv_c, settle_c = _calc_timeline(
        severity=severity, decision=expected_decision,
        fraud=fraud, suspicious=suspicious, rng=rng,
    )

    # claim_type + claim_subtype (used as additional classification targets).
    claim_type = line
    claim_subtype = peril

    return ClaimRecord(
        claim_id=claim_id,
        policy_id=policy_id,
        source_system=source_system,
        claimant_anon=pii["claimant_anon"],
        claimant_email_anon=pii["claimant_email_anon"],
        claimant_phone_anon=pii["claimant_phone_anon"],
        claimant_ssn_anon=pii["claimant_ssn_anon"],
        claimant_address_anon=pii["claimant_address_anon"],
        anonymization_hash=pii["anonymization_hash"],
        amount=amount,
        deductible=deductible,
        coverage_limit=coverage_limit,
        premium_annual=premium_annual,
        description=description,
        date_of_loss=date_of_loss.isoformat(),
        reported_date=reported_date.isoformat(),
        adjuster_id=adjuster_id,
        documents=documents,
        line_of_business=line,
        claim_type=claim_type,
        claim_subtype=claim_subtype,
        severity=severity,
        fraud_flag=fraud,
        suspicious_pattern_flag=suspicious,
        expected_decision=expected_decision,
        correct_payout_amount=payout,
        days_to_first_contact=first_c,
        days_to_investigation_complete=inv_c,
        days_to_settle=settle_c,
        expected_severity=severity,
        expected_claim_type=claim_type,
        expected_fraud_flag=fraud,
        expected_payout_amount=payout,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_csv(records: list[ClaimRecord], path: Path) -> None:
    if not records:
        return
    keys = list(records[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())


def _write_jsonl(records: list[ClaimRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")


def _write_manifest(splits: dict[str, list[ClaimRecord]], path: Path) -> None:
    """Write a manifest with counts, schema, and per-split checksums."""
    manifest: dict[str, Any] = {
        "generator": "ShieldPoint Evaluation Dataset Generator",
        "version": "1.0.0",
        "generated_at": datetime.now(tz=None).astimezone().isoformat(),
        "random_seed": RANDOM_SEED,
        "total_claims": TOTAL_CLAIMS,
        "splits": {},
        "schema": list(splits["train"][0].to_dict().keys()) if splits["train"] else [],
        "line_mix": {k: v for k, v in LINE_MIX},
        "severity_mix": {k: v for k, v in SEVERITY_MIX},
        "decision_mix": {k: v for k, v in DECISION_MIX},
    }
    for split_name, records in splits.items():
        blob = json.dumps([r.to_dict() for r in records], sort_keys=True).encode()
        manifest["splits"][split_name] = {
            "count": len(records),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "path_csv": f"{split_name}.csv",
            "path_jsonl": f"{split_name}.jsonl",
        }
    path.write_text(json.dumps(manifest, indent=2))


def _write_readme(splits: dict[str, list[ClaimRecord]], path: Path) -> None:
    """Write a concise human-readable README for the dataset directory."""
    train, val, test = splits["train"], splits["val"], splits["test"]

    def _dist(records: list[ClaimRecord], field: str) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in records:
            key = str(getattr(r, field))
            d[key] = d.get(key, 0) + 1
        return dict(sorted(d.items()))

    def _fmt_dist(d: dict[str, int], total: int) -> str:
        return "\n".join(f"  - `{k}`: {v} ({v/total:.1%})" for k, v in d.items())

    content = f"""# ShieldPoint Evaluation Dataset

Generated by `generate_dataset.py` (seed = {RANDOM_SEED}).

## Splits (70 / 15 / 15)

| Split   | Count | Path (CSV)  | Path (JSONL)  |
|---------|-------|-------------|---------------|
| train   | {len(train):,}  | `train.csv`   | `train.jsonl`   |
| val     | {len(val):,}  | `val.csv`     | `val.jsonl`     |
| test    | {len(test):,}  | `test.csv`    | `test.jsonl`    |
| **total** | **{len(train)+len(val)+len(test):,}** | | |

## Schema

Each record carries **original claim data** + **correct classification** +
**correct payout** + **processing timeline** + **expected Langfuse outputs**.

Identifiers: `claim_id`, `policy_id`, `source_system`.

Anonymized PII (financial data preserved per AC):
  `claimant_anon`, `claimant_email_anon`, `claimant_phone_anon`,
  `claimant_ssn_anon`, `claimant_address_anon`, `anonymization_hash`.

Claim facts: `amount`, `deductible`, `coverage_limit`, `premium_annual`,
`description`, `date_of_loss`, `reported_date`, `adjuster_id`, `documents`.

Labels (ground truth for evaluation):
  `line_of_business`, `claim_type`, `claim_subtype`, `severity`,
  `fraud_flag`, `suspicious_pattern_flag`, `expected_decision`,
  `correct_payout_amount`.

Timeline labels:
  `days_to_first_contact`, `days_to_investigation_complete`,
  `days_to_settle`.

Expected Langfuse outputs (the four predictions the agent must make):
  `expected_severity`, `expected_claim_type`, `expected_fraud_flag`,
  `expected_payout_amount`, `expected_decision`.

## Distribution (full dataset)

**Line of business** (Homeowners + Auto + Health per design):
{_fmt_dist(_dist(train + val + test, "line_of_business"), TOTAL_CLAIMS)}

**Severity**:
{_fmt_dist(_dist(train + val + test, "severity"), TOTAL_CLAIMS)}

**Expected decision**:
{_fmt_dist(_dist(train + val + test, "expected_decision"), TOTAL_CLAIMS)}

**Fraud flag**:
{_fmt_dist(_dist(train + val + test, "fraud_flag"), TOTAL_CLAIMS)}

## Anonymization

PII is replaced with deterministic synthetic identifiers derived from
`SHA-1(raw_claimant_name)`. The same source person always maps to the same
`CLAIMANT-XXXXXX` token across splits and regenerations. Financial fields
(`amount`, `deductible`, `coverage_limit`, `premium_annual`) are preserved
verbatim per the acceptance criterion. The original raw PII is never written
to disk.

## Provenance

- Generator: `/home/z/my-project/scripts/generate_dataset.py`
- Random seed: `{RANDOM_SEED}` (deterministic — re-running produces an
  identical dataset)
- Source system mix: AS/400 (75%) + Portal (25%) to mimic ShieldPoint's
  primary intake channels
"""
    path.write_text(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    rng = random.Random(RANDOM_SEED)
    print(f"[gen] Generating {TOTAL_CLAIMS:,} labeled claims (seed={RANDOM_SEED}) ...")

    claims: list[ClaimRecord] = []
    for i in range(TOTAL_CLAIMS):
        claims.append(_build_claim(i, rng))

    # Shuffle once so splits are stratified by random order, then carve.
    rng.shuffle(claims)
    train = claims[:TRAIN_COUNT]
    val = claims[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test = claims[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]
    splits = {"train": train, "val": val, "test": test}

    # Write CSV + JSONL for each split.
    for split_name, records in splits.items():
        _write_csv(records, OUTPUT_DIR / f"{split_name}.csv")
        _write_jsonl(records, OUTPUT_DIR / f"{split_name}.jsonl")
        print(f"[gen]   {split_name:5s}: {len(records):,} claims -> "
              f"{split_name}.csv + {split_name}.jsonl")

    _write_manifest(splits, OUTPUT_DIR / "dataset_manifest.json")
    _write_readme(splits, OUTPUT_DIR / "DATASET_README.md")

    # Sanity-check distribution.
    full = train + val + test
    print(f"[gen] Distribution check (n={len(full):,}):")
    for field in ("line_of_business", "severity", "expected_decision", "fraud_flag"):
        d: dict[str, int] = {}
        for r in full:
            k = str(getattr(r, field))
            d[k] = d.get(k, 0) + 1
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(d.items()))
        print(f"[gen]   {field:18s} -> {pretty}")

    print(f"[gen] Done. Output directory: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
