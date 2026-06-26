"""
200+ sample claim input variations for SP-301 unit tests.

Each variation is a tuple ``(input, expected_fields)`` where:
- ``input`` is either a dict (web/email/fax submission) or a string (raw OCR text).
- ``expected_fields`` is a dict of the canonical field values the pipeline
  should produce after extraction + normalisation.

The variations are deliberately diverse to exercise every code path:
- Multiple channels (web, email, fax, unknown).
- Every date format supported by :class:`DateNormalizer`.
- Every currency format supported by :class:`CurrencyNormalizer`.
- Every address abbreviation supported by :class:`AddressNormalizer`.
- Every claim type in the IntakeAgent enum.
- Missing required fields (for the completeness validator).
- Edge cases: empty strings, None, malformed inputs, negative amounts.
"""

from __future__ import annotations

# Each tuple: (input, expected_subset)
# `expected_subset` only contains the fields we assert on — the pipeline
# may extract more, but these MUST be present and equal.
SAMPLE_CLAIMS: list[tuple[object, dict]] = [
    # ----------------------------------------------------------------
    # 1-30: Web portal submissions (dict input, source=web)
    # ----------------------------------------------------------------
    ({"source": "web", "policyholder_name": "Alice Homeowner", "policy_id": "HO-2024-001",
      "claim_type": "homeowners", "date_of_loss": "2026-03-14",
      "damage_description": "Wind damage to roof shingles.",
      "amount_claimed": 1250.00, "incident_location": "123 Main St Springfield IL 62704",
      "phone": "(555) 123-4567", "email": "alice@example.com"},
     {"policyholder_name": "Alice Homeowner", "policy_id": "HO-2024-001",
      "claim_type": "homeowners", "date_of_loss": "2026-03-14",
      "amount_claimed": 1250.00}),

    ({"source": "web", "policyholder_name": "Bob Driver", "policy_id": "AU-2024-015",
      "claim_type": "auto", "date_of_loss": "04/02/2026",
      "damage_description": "Rear-end collision damage.",
      "amount_claimed": "$4,800.00"},
     {"policyholder_name": "Bob Driver", "policy_id": "AU-2024-015",
      "claim_type": "auto", "date_of_loss": "2026-04-02", "amount_claimed": 4800.00}),

    ({"source": "web", "policyholder_name": "Carol Resident", "policy_id": "HO-2024-088",
      "claim_type": "homeowners", "date_of_loss": "5/10/26",
      "damage_description": "Hail damage to mailbox.",
      "amount_claimed": "250"},
     {"policyholder_name": "Carol Resident", "policy_id": "HO-2024-088",
      "claim_type": "homeowners", "date_of_loss": "2026-05-10", "amount_claimed": 250.00}),

    ({"source": "web", "policyholder_name": "Dan Property", "policy_id": "HO-2024-012",
      "claim_type": "homeowners", "date_of_loss": "February 28, 2026",
      "damage_description": "Flood damage to basement.",
      "amount_claimed": "12,500.00"},
     {"policyholder_name": "Dan Property", "policy_id": "HO-2024-012",
      "claim_type": "homeowners", "date_of_loss": "2026-02-28", "amount_claimed": 12500.00}),

    ({"source": "web", "policyholder_name": "Eve Merchant", "policy_id": "GL-2024-001",
      "claim_type": "property", "date_of_loss": "15 March 2026",
      "damage_description": "Warehouse fire damage.",
      "amount_claimed": "USD 50000"},
     {"policyholder_name": "Eve Merchant", "policy_id": "GL-2024-001",
      "claim_type": "property", "date_of_loss": "2026-03-15", "amount_claimed": 50000.00}),

    ({"source": "web", "policyholder_name": "Frank Liability", "policy_id": "GL-2024-002",
      "claim_type": "liability", "date_of_loss": "2026-06-01",
      "damage_description": "Customer slip and fall injury.",
      "amount_claimed": 8000.50},
     {"policyholder_name": "Frank Liability", "policy_id": "GL-2024-002",
      "claim_type": "liability", "date_of_loss": "2026-06-01", "amount_claimed": 8000.50}),

    ({"source": "web", "policyholder_name": "Grace Health", "policy_id": "HE-2024-100",
      "claim_type": "health", "date_of_loss": "12-25-2025",
      "damage_description": "Emergency room visit after fall.",
      "amount_claimed": "1,234.56"},
     {"policyholder_name": "Grace Health", "policy_id": "HE-2024-100",
      "claim_type": "health", "date_of_loss": "2025-12-25", "amount_claimed": 1234.56}),

    ({"source": "web", "policyholder_name": "Henry Auto", "policy_id": "AU-2024-020",
      "claim_type": "auto", "date_of_loss": "Sept 1, 2026",
      "damage_description": "Hail damage to vehicle.",
      "amount_claimed": "$2,500.00"},
     {"policyholder_name": "Henry Auto", "policy_id": "AU-2024-020",
      "claim_type": "auto", "date_of_loss": "2026-09-01", "amount_claimed": 2500.00}),

    ({"source": "web", "policyholder_name": "Ivy Property", "policy_id": "HO-2024-200",
      "claim_type": "homeowners", "date_of_loss": "2026-07-04",
      "damage_description": "Fire damage to kitchen.",
      "amount_claimed": "15000",
      "incident_location": "456 Oak Ave, Beverly Hills CA 90210"},
     {"policyholder_name": "Ivy Property", "policy_id": "HO-2024-200",
      "claim_type": "homeowners", "date_of_loss": "2026-07-04",
      "amount_claimed": 15000.00}),

    ({"source": "web", "policyholder_name": "Jack Homeowner", "policy_id": "HO-2024-300",
      "claim_type": "homeowners", "date_of_loss": "11/30/2025",
      "damage_description": "Burst pipe water damage.",
      "amount_claimed": 8750.25},
     {"policyholder_name": "Jack Homeowner", "policy_id": "HO-2024-300",
      "claim_type": "homeowners", "date_of_loss": "2025-11-30", "amount_claimed": 8750.25}),

    # 11-20: more web variations
    *[
        ({"source": "web", "policyholder_name": f"Test User {i}",
          "policy_id": f"HO-2024-{i:04d}", "claim_type": "homeowners",
          "date_of_loss": f"2026-01-{i:02d}",
          "damage_description": f"Test damage description {i}.",
          "amount_claimed": float(i * 100)},
         {"policyholder_name": f"Test User {i}", "policy_id": f"HO-2024-{i:04d}",
          "claim_type": "homeowners", "date_of_loss": f"2026-01-{i:02d}",
          "amount_claimed": float(i * 100)})
        for i in range(11, 21)
    ],

    # 21-30: web with EU date formats and currency
    *[
        ({"source": "web", "policyholder_name": f"EU User {i}",
          "policy_id": f"HO-2024-{i:04d}", "claim_type": "homeowners",
          "date_of_loss": f"{i:02d}/01/2026",
          "damage_description": f"EU date test {i}.",
          "amount_claimed": f"1.{i:03d},50"},
         {"policyholder_name": f"EU User {i}", "policy_id": f"HO-2024-{i:04d}",
          "claim_type": "homeowners",
          "amount_claimed": 1000.0 + i + 0.50})
        for i in range(21, 31)
    ],

    # ----------------------------------------------------------------
    # 31-70: Raw text inputs simulating fax OCR / email body
    # ----------------------------------------------------------------
    ("Policy ID: HO-2024-001\nPolicyholder Name: Alice Homeowner\nDate of Loss: 2026-03-14\nClaim Type: homeowners\nDescription: Wind damage to roof shingles during storm.\nAmount: $1,250.00\nLocation: 123 Main St Springfield IL 62704\nPhone: (555) 123-4567\nEmail: alice@example.com",
     {"policyholder_name": "Alice Homeowner", "policy_id": "HO-2024-001",
      "claim_type": "homeowners", "date_of_loss": "2026-03-14", "amount_claimed": 1250.00}),

    ("Policy No.: AU-2024-015\nInsured: Bob Driver\nD.O.L.: 04/02/2026\nClaim Type: auto\nDamage: Rear-end collision damage.\nAmount Claimed: $4,800.00",
     {"policyholder_name": "Bob Driver", "policy_id": "AU-2024-015",
      "claim_type": "auto", "date_of_loss": "2026-04-02", "amount_claimed": 4800.00}),

    ("POLICY ID: HO-2024-088\nNAME: Carol Resident\nLOSS DATE: 5/10/26\nCLAIM TYPE: homeowners\nDESCRIPTION: Hail damage to mailbox.\nDAMAGES: 250",
     {"policyholder_name": "Carol Resident", "policy_id": "HO-2024-088",
      "claim_type": "homeowners", "date_of_loss": "2026-05-10", "amount_claimed": 250.00}),

    ("Policy: HO-2024-012\nPolicyholder: Dan Property\nDate of Loss: February 28, 2026\nClaim Type: homeowners\nDescription: Flood damage to basement.\nAmount: 12,500.00",
     {"policyholder_name": "Dan Property", "policy_id": "HO-2024-012",
      "claim_type": "homeowners", "date_of_loss": "2026-02-28", "amount_claimed": 12500.00}),

    ("From: claims@example.com\nSubject: New Claim\nPolicy ID: GL-2024-001\nPolicyholder Name: Eve Merchant\nDate of Loss: 15 March 2026\nClaim Type: property\nDescription: Warehouse fire damage.\nAmount: USD 50000",
     {"policyholder_name": "Eve Merchant", "policy_id": "GL-2024-001",
      "claim_type": "property", "date_of_loss": "2026-03-15", "amount_claimed": 50000.00}),

    ("Fax received 2026-06-02\nPolicy ID: GL-2024-002\nInsured Name: Frank Liability\nLoss Date: 2026-06-01\nClaim Type: liability\nDescription: Customer slip and fall injury.\nDamages: $8,000.50",
     {"policyholder_name": "Frank Liability", "policy_id": "GL-2024-002",
      "claim_type": "liability", "date_of_loss": "2026-06-01", "amount_claimed": 8000.50}),

    # 37-50: text with bare policy IDs (no "Policy ID:" prefix)
    *[
        (f"Claim report {i}. Policy HO-2024-{i:04d}. Insured: Test Claimant {i}. "
         f"Date of loss: 2026-06-{i:02d}. Damage: test damage number {i}. Amount: ${i * 100}.00",
         {"policy_id": f"HO-2024-{i:04d}", "amount_claimed": float(i * 100)})
        for i in range(37, 51)
    ],

    # 51-70: text with various date formats (days > 12 to avoid US/EU ambiguity;
    # uses July which has 31 days so all of i+12 for i in 1..19 are valid)
    *[
        (f"Policy HO-2024-{i:04d}. Insured: Date Test {i}. "
         f"Date of loss: {i + 12:02d}/07/2026. Damage: date test {i}.",
         {"policy_id": f"HO-2024-{i:04d}", "date_of_loss": f"2026-07-{i + 12:02d}"})
        for i in range(1, 20)  # produces 19 entries (claims 51-69)
    ],

    # ----------------------------------------------------------------
    # 71-110: Email channel — dict with sender/attachments
    # ----------------------------------------------------------------
    *[
        ({"source": "email", "sender": f"user{i}@example.com",
          "received_at": f"2026-06-{i:02d}T10:00:00Z",
          "subject": f"Claim submission {i}",
          "policyholder_name": f"Email User {i}",
          "policy_id": f"HO-2024-{i:04d}",
          "claim_type": "homeowners",
          "date_of_loss": f"2026-06-{i:02d}",
          "damage_description": f"Email-submitted damage {i}.",
          "amount_claimed": float(i * 50)},
         {"policyholder_name": f"Email User {i}", "policy_id": f"HO-2024-{i:04d}",
          "claim_type": "homeowners", "amount_claimed": float(i * 50)})
        for i in range(71, 111)
    ],

    # ----------------------------------------------------------------
    # 111-150: Fax channel — dict with pdf_bytes
    # ----------------------------------------------------------------
    *[
        ({"source": "fax", "fax_number": f"+1-555-{i:04d}",
          "received_at": f"2026-07-{i % 30 + 1:02d}T14:00:00Z",
          "subject": f"Fax claim {i}",
          "policyholder_name": f"Fax User {i}",
          "policy_id": f"AU-2024-{i:04d}",
          "claim_type": "auto",
          "date_of_loss": f"2026-07-{i % 30 + 1:02d}",
          "damage_description": f"Fax-submitted damage {i}.",
          "amount_claimed": float(i * 25)},
         {"policyholder_name": f"Fax User {i}", "policy_id": f"AU-2024-{i:04d}",
          "claim_type": "auto", "amount_claimed": float(i * 25)})
        for i in range(111, 151)
    ],

    # ----------------------------------------------------------------
    # 151-180: Currency edge cases
    # ----------------------------------------------------------------
    *[
        ({"source": "web", "policyholder_name": f"Currency Test {i}",
          "policy_id": f"HO-2024-{i:04d}", "claim_type": "homeowners",
          "date_of_loss": "2026-08-01",
          "damage_description": f"Currency edge case {i}.",
          "amount_claimed": amt},
         {"policyholder_name": f"Currency Test {i}", "policy_id": f"HO-2024-{i:04d}",
          "amount_claimed": expected})
        for i, (amt, expected) in enumerate([
            ("$0.99", 0.99), ("$1,234,567.89", 1234567.89),
            ("0", 0.0), ("$0", 0.0), ("USD 0.00", 0.0),
            ("1,000.00", 1000.00), ("1000.00", 1000.00),
            ("$10.00", 10.00), ("10.00", 10.00),
            ("$100.00", 100.00), ("100.00", 100.00),
            ("$1,000.00", 1000.00), ("1,000.00", 1000.00),
            ("$10,000.00", 10000.00), ("10,000.00", 10000.00),
            ("$100,000.00", 100000.00), ("100,000.00", 100000.00),
            ("$1,000,000.00", 1000000.00), ("1,000,000.00", 1000000.00),
            ("$1,234,567.89", 1234567.89), ("1,234,567.89", 1234567.89),
            ("1.234,56", 1234.56), ("1234,56", 1234.56),
            ("$0.01", 0.01), ("0.01", 0.01),
            ("$999.99", 999.99), ("999.99", 999.99),
            ("$1000000.00", 1000000.00), ("1000000", 1000000.0),
        ], start=151)
    ],

    # ----------------------------------------------------------------
    # 181-200: Address edge cases
    # ----------------------------------------------------------------
    *[
        ({"source": "web", "policyholder_name": f"Address Test {i}",
          "policy_id": f"HO-2024-{i:04d}", "claim_type": "homeowners",
          "date_of_loss": "2026-09-01",
          "damage_description": f"Address edge case {i}.",
          "amount_claimed": 1000.00,
          "incident_location": addr},
         {"policyholder_name": f"Address Test {i}", "policy_id": f"HO-2024-{i:04d}"})
        for i, addr in enumerate([
            "123 main st", "456 ELM AVE", "789 OAK BLVD",
            "1 PARK PL", "2 DR", "3 LN", "4 CT", "5 RD",
            "6 STE 100", "7 APT 2B", "8 N Main St", "9 S Elm Ave",
            "10 NE Oak Blvd", "11 SW Park Pl", "12 NW Dr",
            "13 SE Ln", "14 CT", "15 PL",
            "123 Main St, Springfield IL 62704",
            "456 Elm Ave, Beverly Hills CA 90210-1234",
        ], start=181)
    ],

    # ----------------------------------------------------------------
    # 201-220: Missing-field variations (for completeness validator)
    # ----------------------------------------------------------------
    # Missing policy_id
    ({"source": "web", "policyholder_name": "Missing PID",
      "claim_type": "homeowners", "date_of_loss": "2026-01-01",
      "damage_description": "Missing policy ID test."},
     {"policyholder_name": "Missing PID"}),

    # Missing policyholder_name
    ({"source": "web", "policy_id": "HO-2024-001",
      "claim_type": "homeowners", "date_of_loss": "2026-01-01",
      "damage_description": "Missing name test."},
     {"policy_id": "HO-2024-001"}),

    # Missing claim_type
    ({"source": "web", "policyholder_name": "Has Name",
      "policy_id": "HO-2024-001", "date_of_loss": "2026-01-01",
      "damage_description": "Missing type test."},
     {"policyholder_name": "Has Name", "policy_id": "HO-2024-001"}),

    # Missing date_of_loss
    ({"source": "web", "policyholder_name": "Has Name",
      "policy_id": "HO-2024-001", "claim_type": "homeowners",
      "damage_description": "Missing date test."},
     {"policyholder_name": "Has Name", "policy_id": "HO-2024-001",
      "claim_type": "homeowners"}),

    # Missing damage_description
    ({"source": "web", "policyholder_name": "Has Name",
      "policy_id": "HO-2024-001", "claim_type": "homeowners",
      "date_of_loss": "2026-01-01"},
     {"policyholder_name": "Has Name", "policy_id": "HO-2024-001",
      "claim_type": "homeowners", "date_of_loss": "2026-01-01"}),

    # Empty strings (should be treated as missing)
    ({"source": "web", "policyholder_name": "", "policy_id": "",
      "claim_type": "", "date_of_loss": "",
      "damage_description": ""},
     {}),

    # Whitespace-only strings (should be treated as missing)
    ({"source": "web", "policyholder_name": "   ", "policy_id": "  ",
      "claim_type": " ", "date_of_loss": "  ",
      "damage_description": "   "},
     {}),

    # None values
    ({"source": "web", "policyholder_name": None, "policy_id": None,
      "claim_type": None, "date_of_loss": None,
      "damage_description": None},
     {}),

    # Multiple missing
    *[
        ({"source": "web",
          "policyholder_name": f"Multi Missing {i}" if i % 2 == 0 else None,
          "policy_id": f"HO-2024-{i:04d}" if i % 3 == 0 else None,
          "claim_type": "homeowners" if i % 4 == 0 else None,
          "date_of_loss": f"2026-01-{i:02d}" if i % 5 == 0 else None,
          "damage_description": f"Multi missing test {i}." if i % 6 == 0 else None},
         {})
        for i in range(1, 13)
    ],

    # ----------------------------------------------------------------
    # 221-240: Claim type inference from description (no explicit claim_type)
    # ----------------------------------------------------------------
    *[
        ({"source": "web", "policyholder_name": f"Infer Type {i}",
          "policy_id": f"HO-2024-{i:04d}",
          "date_of_loss": "2026-10-01",
          "damage_description": desc,
          "amount_claimed": 1000.00},
         {"policy_id": f"HO-2024-{i:04d}"})
        for i, desc in enumerate([
            "Car accident on highway.", "Home roof damage.",
            "Commercial building fire.", "Customer slip and fall injury.",
            "Hospital emergency room visit.", "Stolen vehicle recovery.",
            "Truck collision on freeway.", "House basement flooding.",
            "Warehouse structural damage.", "Negligence claim by tenant.",
            "Medical treatment required.", "Burglary at residence.",
            "Auto windshield cracked.", "Kitchen fire damage.",
            "Office building water damage.", "Bodily injury from fall.",
            "Health insurance claim.", "Robbery at retail store.",
            "Vehicle theft report.", "Homeowners insurance claim.",
        ], start=221)
    ],
]

# Sanity check at import time — make sure we actually have 200+ entries.
assert len(SAMPLE_CLAIMS) >= 200, (
    f"SAMPLE_CLAIMS has only {len(SAMPLE_CLAIMS)} entries — need 200+"
)
