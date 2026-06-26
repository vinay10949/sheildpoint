#!/usr/bin/env python3
"""
SP-205 — Prepare Training Data for Qwen3.6 LoRA Fine-Tuning
=============================================================

Converts the historical claims dataset (from SP-204) into the
instruction-following format required for LoRA fine-tuning.

The input is the train.jsonl file containing 3,500 historical claims
with labels (severity, claim_type, fraud_risk_score). The output is a
JSONL file of instruction-following examples suitable for QLoRA training
with the Qwen3.6 35B A3B model.

Format
------
Each line in the output JSONL is::

    {
        "instruction": "<system prompt explaining the classification task>",
        "input": "<JSON-encoded claim data + claimant history>",
        "output": "<JSON-encoded classification result>"
    }

This matches the Alpaca instruction format, which is supported by the
PEFT/TRL training pipeline used in ``lora_train.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shieldpoint.training.prepare")

# The system prompt matches the ClassifierAgent's SYSTEM_PROMPT in agents.py
SYSTEM_PROMPT = """\
You are the ClassifierAgent in the ShieldPoint claims automation system.

Your job: classify the claim along three dimensions and return a JSON object.

Dimensions:
1. severity: "low" | "medium" | "high"
   - low:    claim amount <= $1,000 OR minor cosmetic damage
   - medium: $1,000 < amount <= $10,000 OR moderate structural damage
   - high:   amount > $10,000 OR total loss / severe structural damage

2. claim_type: one of:
   - "property_damage", "auto", "liability", "medical", "water_damage",
     "theft", "vandalism", "fire", "wind", "hail"

3. fraud_risk_score: float in [0.0, 1.0]
   - 0.0 = clearly legitimate
   - 0.3 = some risk indicators but plausibly legitimate
   - 0.6 = significant fraud indicators
   - 1.0 = clear fraud

Return JSON of this exact shape:
{
  "severity": "low" | "medium" | "high",
  "claim_type": "<one of the values above>",
  "fraud_risk_score": <float in [0.0, 1.0]>,
  "confidence": <float in [0.0, 1.0]>,
  "reasoning": "<2-3 sentences explaining your assessment>",
  "fraud_indicators": ["<indicator 1>", "<indicator 2>", ...],
  "ambiguous": true | false,
  "ambiguity_reason": "<only if ambiguous=true>"
}
"""


def build_instruction_example(claim: dict[str, Any]) -> dict[str, str]:
    """Build a single instruction-following example from a labeled claim."""
    # The "input" is the claim data + claimant history
    input_data = {
        "claim": {
            "claim_id": claim.get("claim_id", "CLM-TRAIN-000"),
            "policy_id": claim.get("policy_id", "POL-000"),
            "claimant": claim.get("claimant", "Unknown"),
            "amount": claim.get("amount", 0),
            "date_of_loss": claim.get("date_of_loss", "2024-01-01"),
            "description": claim.get("description", claim.get("damage_description", "")),
            "claim_type_hint": claim.get("claim_type_hint", claim.get("claim_type", "")),
        },
        "claimant_history": {
            "prior_claims_count": claim.get("prior_claims_count", 0),
            "avg_prior_claim_amount": claim.get("avg_prior_claim_amount", 0),
            "days_since_last_claim": claim.get("days_since_last_claim", None),
            "policy_inception_days_ago": claim.get("policy_inception_days_ago", 365),
        },
    }

    # The "output" is the expected classification result
    output_data = {
        "severity": claim.get("label_severity", claim.get("severity", "low")),
        "claim_type": claim.get("label_claim_type", claim.get("claim_type", "property_damage")),
        "fraud_risk_score": claim.get("label_fraud_risk_score",
                                       claim.get("fraud_risk_score", 0.1)),
        "confidence": claim.get("label_confidence", 0.9),
        "reasoning": claim.get("label_reasoning",
                                "Claim classified based on amount, peril, and claimant history."),
        "fraud_indicators": claim.get("label_fraud_indicators", []),
        "ambiguous": claim.get("label_ambiguous", False),
        "ambiguity_reason": claim.get("label_ambiguity_reason", ""),
    }

    return {
        "instruction": SYSTEM_PROMPT,
        "input": json.dumps(input_data, indent=2),
        "output": json.dumps(output_data, indent=2),
    }


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Prepare the instruction-following dataset from historical claims.

    Parameters
    ----------
    input_path : Path
        Path to the train.jsonl file from SP-204 (3,500 claims).
    output_path : Path
        Directory to write the train/val/test splits.
    train_ratio, val_ratio : float
        Train/validation/test split ratios (test = 1 - train - val).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Counts of examples in each split.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Read all claims
    claims: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    logger.info("Loaded %d claims from %s", len(claims), input_path)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(claims)
    n = len(claims)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_claims = claims[:n_train]
    val_claims = claims[n_train:n_train + n_val]
    test_claims = claims[n_train + n_val:]

    # Convert to instruction format and write
    counts = {}
    for split_name, split_claims in [
        ("train", train_claims),
        ("val", val_claims),
        ("test", test_claims),
    ]:
        split_path = output_path / f"{split_name}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for claim in split_claims:
                example = build_instruction_example(claim)
                f.write(json.dumps(example) + "\n")
        counts[split_name] = len(split_claims)
        logger.info("Wrote %d examples to %s", len(split_claims), split_path)

    # Write a manifest
    manifest = {
        "source": str(input_path),
        "total_claims": n,
        "splits": counts,
        "seed": seed,
        "format": "alpaca_instruction",
        "model_target": "qwen3.6-35b-a3b",
    }
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare instruction-following training data for LoRA fine-tuning"
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("/home/z/my-project/sheildpoint/download/dataset/train.jsonl"),
        help="Path to the historical claims train.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/data"),
        help="Output directory for the instruction-format splits",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    counts = prepare_dataset(args.input, args.output, seed=args.seed)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
