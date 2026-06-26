#!/usr/bin/env python3
"""
ShieldPoint Langfuse Evaluation Dataset Uploader
=================================================

Uploads the 750-claim test set to a Langfuse project as an evaluation
dataset, ready to be used by the automated evaluation script after each
agent deployment.

This script:
  1. Reads /home/z/my-project/download/dataset/test.jsonl
  2. Builds a Langfuse datasetitem per claim. The `input` is the claim
     record (the agent's prompt context). The `expected_output` is the
     ground-truth bundle (severity, claim_type, fraud_flag, payout,
     decision).
  3. Calls the Langfuse Python SDK to create (or update) a dataset named
     "shieldpoint-claims-test-v1" with all 750 items.
  4. Writes a manifest of uploaded item IDs to
     /home/z/my-project/download/langfuse/upload_manifest.json.

Usage
-----
  # Verify the payload shape without sending anything:
  python3 upload_to_langfuse.py --dry-run --limit 3

  # Push all 750 items to a local Langfuse instance:
  python3 upload_to_langfuse.py \\
      --host http://localhost:3000 \\
      --public-key pk-lf-... \\
      --secret-key sk-lf-...

Environment fallbacks (read if flags omitted):
  LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

Idempotency
-----------
Re-running with the same dataset name will add items again. To re-upload
cleanly, pass --reset to delete the existing dataset first (requires the
SDK's `delete_dataset` capability, available in Langfuse v3+).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

TEST_SET_PATH = Path("/home/z/my-project/download/dataset/test.jsonl")
OUTPUT_DIR = Path("/home/z/my-project/download/langfuse")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATASET_NAME = "shieldpoint-claims-test-v1"


def build_datasetitem(claim: dict[str, Any]) -> dict[str, Any]:
    """Convert a single claim record into a Langfuse datasetitem payload.

    The `input` is the claim context the agent sees (everything EXCEPT the
    expected_* labels -- those go in expected_output). The `expected_output`
    is the ground-truth bundle the agent must match.

    `metadata` carries provenance for traceability inside Langfuse.
    """
    input_fields = [
        "claim_id", "policy_id", "source_system",
        "claimant_anon", "claimant_email_anon", "claimant_phone_anon",
        "claimant_ssn_anon", "claimant_address_anon",
        "amount", "deductible", "coverage_limit", "premium_annual",
        "description", "date_of_loss", "reported_date",
        "adjuster_id", "documents",
        "line_of_business", "claim_subtype",  # claim_subtype is given; severity is predicted
    ]
    expected_output = {
        "decision":            claim["expected_decision"],
        "severity":            claim["expected_severity"],
        "claim_type":          claim["expected_claim_type"],
        "fraud_flag":          claim["expected_fraud_flag"],
        "payout_amount":       claim["expected_payout_amount"],
        "days_to_settle":      claim["days_to_settle"],
    }
    item = {
        "input": {k: claim[k] for k in input_fields},
        "expected_output": expected_output,
        "metadata": {
            "claim_id": claim["claim_id"],
            "source_system": claim["source_system"],
            "anonymization_hash": claim["anonymization_hash"],
            "fraud_flag": claim["fraud_flag"],
            "suspicious_pattern_flag": claim["suspicious_pattern_flag"],
            "uploaded_by": "upload_to_langfuse.py",
            "uploaded_at": datetime.now().astimezone().isoformat(),
        },
    }
    return item


def upload(items: list[dict[str, Any]], dataset_name: str, host: str,
           public_key: str, secret_key: str) -> dict[str, Any]:
    """Push items to Langfuse via the Python SDK. Returns a manifest."""
    try:
        from langfuse import Langfuse  # type: ignore
    except ImportError:
        print("ERROR: langfuse SDK not installed. Run: pip install langfuse",
              file=sys.stderr)
        sys.exit(2)

    lf = Langfuse(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
    )

    # Create or fetch the dataset.
    try:
        dataset = lf.get_dataset(dataset_name)
        print(f"[upload] Reusing existing dataset '{dataset_name}' "
              f"(id={getattr(dataset, 'id', '?')})")
    except Exception:
        # SDK raises if dataset doesn't exist; create it.
        try:
            dataset = lf.create_dataset(
                name=dataset_name,
                description=(
                    "ShieldPoint 750-claim test set for agent regression "
                    "evaluation. Generated from "
                    "/home/z/my-project/download/dataset/test.jsonl"
                ),
            )
            print(f"[upload] Created dataset '{dataset_name}'")
        except Exception as exc:
            print(f"[upload] ERROR creating dataset: {exc}", file=sys.stderr)
            sys.exit(3)

    item_ids: list[str] = []
    for i, item in enumerate(items, 1):
        try:
            created = lf.create_dataset_item(
                dataset_name=dataset_name,
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item["metadata"],
            )
            item_ids.append(getattr(created, "id", f"item-{i:04d}"))
            if i % 50 == 0:
                print(f"[upload]   {i}/{len(items)} items uploaded")
        except Exception as exc:
            print(f"[upload] WARN item {i} failed: {exc}", file=sys.stderr)

    return {
        "dataset_name": dataset_name,
        "host": host,
        "uploaded_at": datetime.now().astimezone().isoformat(),
        "total_items": len(items),
        "successful_uploads": len(item_ids),
        "item_ids": item_ids,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test-set", type=Path, default=TEST_SET_PATH,
                   help=f"Path to test.jsonl (default: {TEST_SET_PATH})")
    p.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME,
                   help=f"Langfuse dataset name (default: {DEFAULT_DATASET_NAME})")
    p.add_argument("--host",
                   default=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"))
    p.add_argument("--public-key",
                   default=os.environ.get("LANGFUSE_PUBLIC_KEY"))
    p.add_argument("--secret-key",
                   default=os.environ.get("LANGFUSE_SECRET_KEY"))
    p.add_argument("--dry-run", action="store_true",
                   help="Print payloads instead of uploading. Verifies shape.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N claims (0 = all). "
                        "Useful with --dry-run.")
    p.add_argument("--reset", action="store_true",
                   help="Delete the existing dataset before re-uploading "
                        "(idempotent re-runs).")
    args = p.parse_args()

    if not args.test_set.exists():
        print(f"ERROR: test set not found at {args.test_set}", file=sys.stderr)
        print("       Run generate_dataset.py first.", file=sys.stderr)
        return 1

    # Load test set.
    claims = []
    with args.test_set.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    if args.limit > 0:
        claims = claims[:args.limit]
    print(f"[upload] Loaded {len(claims)} claims from {args.test_set}")

    items = [build_datasetitem(c) for c in claims]

    # Dry-run mode: print first 3 items, write all to file, exit.
    if args.dry_run:
        sample_path = OUTPUT_DIR / "dry_run_sample.json"
        sample_path.write_text(json.dumps(items[:3], indent=2))
        full_path = OUTPUT_DIR / "dry_run_full.jsonl"
        with full_path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"[upload] DRY RUN -- wrote sample to {sample_path}")
        print(f"[upload] DRY RUN -- wrote full payload to {full_path}")
        print(f"[upload] First item preview:")
        print(json.dumps(items[0], indent=2)[:1500])
        return 0

    # Real upload requires credentials.
    if not args.public_key or not args.secret_key:
        print("ERROR: --public-key and --secret-key required for real upload.",
              file=sys.stderr)
        print("       (Or set $LANGFUSE_PUBLIC_KEY / $LANGFUSE_SECRET_KEY)",
              file=sys.stderr)
        print("       Use --dry-run to inspect payloads without uploading.",
              file=sys.stderr)
        return 1

    if args.reset:
        try:
            from langfuse import Langfuse  # type: ignore
            lf = Langfuse(host=args.host, public_key=args.public_key,
                          secret_key=args.secret_key)
            try:
                lf.delete_dataset(args.dataset_name)
                print(f"[upload] Deleted existing dataset '{args.dataset_name}'")
            except Exception as exc:
                print(f"[upload] --reset: could not delete (may not exist): {exc}",
                      file=sys.stderr)
        except ImportError:
            print("ERROR: langfuse SDK not installed.", file=sys.stderr)
            return 2

    manifest = upload(items, args.dataset_name, args.host,
                      args.public_key, args.secret_key)
    manifest_path = OUTPUT_DIR / "upload_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[upload] Done. {manifest['successful_uploads']}/{manifest['total_items']} "
          f"items uploaded to dataset '{args.dataset_name}'.")
    print(f"[upload] Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
