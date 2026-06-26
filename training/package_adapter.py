#!/usr/bin/env python3
"""
SP-205 — Package LoRA Adapter for LM Studio
=============================================

Packages the fine-tuned LoRA adapter for loading alongside the base model
in LM Studio. LM Studio supports loading LoRA adapters via the
``--lora`` flag or the adapter directory convention.

This script:
1. Copies the adapter weights to a standard directory structure.
2. Generates an ``adapter_config.json`` with the metadata LM Studio needs.
3. Creates a ``README.md`` documenting the adapter's purpose and metrics.
4. Optionally zips the adapter for distribution.

Output structure::

    adapters/qwen3.6-35b-a3b-shieldpoint-v1/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── README.md
    └── metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shieldpoint.training.package")


def package_adapter(
    *,
    adapter_source: Path,
    output_dir: Path,
    adapter_name: str,
    base_model: str = "Qwen/Qwen3.6-35B-A3B",
    metrics: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    create_zip: bool = True,
) -> Path:
    """Package the LoRA adapter for LM Studio loading.

    Parameters
    ----------
    adapter_source : Path
        Directory containing the trained adapter weights (from lora_train.py).
    output_dir : Path
        Parent directory for the packaged adapter.
    adapter_name : str
        Name for the adapter (e.g. "qwen3.6-35b-a3b-shieldpoint-v1").
    base_model : str
        HuggingFace ID of the base model this adapter was trained on.
    metrics : dict, optional
        Evaluation metrics (from evaluate_adapter.py).
    training_summary : dict, optional
        Training summary (from lora_train.py).
    create_zip : bool
        If True, also create a zip archive for distribution.

    Returns
    -------
    Path
        Path to the packaged adapter directory.
    """
    package_dir = output_dir / adapter_name
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy adapter weights and config
    logger.info("Copying adapter files from %s to %s", adapter_source, package_dir)
    for item in adapter_source.iterdir():
        if item.is_file():
            shutil.copy2(item, package_dir / item.name)

    # Generate or update adapter_config.json
    config_path = package_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    config.update({
        "base_model_name_or_path": base_model,
        "adapter_name": adapter_name,
        "trained_for": "shieldpoint_claims_classification",
        "peft_type": "LORA",
        "r": training_summary.get("rank", 16) if training_summary else 16,
        "lora_alpha": training_summary.get("alpha", 32) if training_summary else 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote adapter_config.json")

    # Save metrics
    if metrics:
        metrics_path = package_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics.json")

    # Generate README
    readme_path = package_dir / "README.md"
    readme_content = _generate_readme(
        adapter_name=adapter_name,
        base_model=base_model,
        metrics=metrics,
        training_summary=training_summary,
    )
    readme_path.write_text(readme_content)
    logger.info("Wrote README.md")

    # Create zip archive
    if create_zip:
        zip_path = output_dir / f"{adapter_name}.zip"
        shutil.make_archive(
            str(output_dir / adapter_name),  # base name (no .zip)
            "zip",
            root_dir=output_dir,
            base_dir=adapter_name,
        )
        logger.info("Created zip archive: %s", zip_path)

    return package_dir


def _generate_readme(
    *,
    adapter_name: str,
    base_model: str,
    metrics: dict[str, Any] | None,
    training_summary: dict[str, Any] | None,
) -> str:
    """Generate a README.md for the adapter package."""
    lines = [
        f"# {adapter_name}",
        "",
        "## Overview",
        "",
        f"- **Base model**: `{base_model}`",
        "- **Adapter type**: LoRA (Low-Rank Adaptation)",
        "- **Task**: ShieldPoint insurance claims classification",
        "- **Target modules**: `q_proj`, `v_proj`",
        f"- **LoRA rank**: {training_summary.get('rank', 16) if training_summary else 16}",
        f"- **LoRA alpha**: {training_summary.get('alpha', 32) if training_summary else 32}",
        "",
        "## Purpose",
        "",
        "This adapter fine-tunes the Qwen3.6 35B A3B base model on ShieldPoint's",
        "historical claims data to improve classification accuracy for insurance",
        "claim severity, type, and fraud risk scoring. The base model's general",
        "reasoning capabilities are preserved (QLoRA approach — base weights are",
        "frozen, only the adapter is trained).",
        "",
        "## Loading in LM Studio",
        "",
        "1. Load the base model `Qwen/Qwen3.6-35B-A3B` in LM Studio.",
        "2. Go to the LoRA adapters tab.",
        f"3. Select this adapter directory (`{adapter_name}/`).",
        "4. Apply the adapter — the model now uses the fine-tuned weights for",
        "   inference while keeping the base model available for other tasks.",
        "",
        "## Metrics",
        "",
    ]

    if metrics:
        lines.extend([
            f"- **Severity accuracy**: {metrics.get('severity_accuracy', 0):.2%}",
            f"- **Claim type accuracy**: {metrics.get('claim_type_accuracy', 0):.2%}",
            f"- **Exact match accuracy**: {metrics.get('exact_match_accuracy', 0):.2%}",
            f"- **Fraud risk MAE**: {metrics.get('fraud_risk_mae', 0):.4f}",
            "",
        ])
        if "improvement_pp" in metrics:
            lines.extend([
                f"- **Improvement over base**: {metrics['improvement_pp']:+.1f}pp",
                f"- **Meets 5pp threshold**: {'Yes' if metrics.get('meets_5pp_threshold') else 'No'}",
                "",
            ])
    else:
        lines.extend([
            "_(No metrics available — run `evaluate_adapter.py` and re-package.)_",
            "",
        ])

    if training_summary:
        lines.extend([
            "## Training Details",
            "",
            f"- **Epochs**: {training_summary.get('epochs', 3)}",
            f"- **Training examples**: {training_summary.get('train_examples', 0)}",
            f"- **Validation examples**: {training_summary.get('val_examples', 0)}",
            f"- **Final training loss**: {training_summary.get('train_loss', 'N/A')}",
            f"- **Final eval loss**: {training_summary.get('eval_loss', 'N/A')}",
            "",
        ])

    lines.extend([
        "## Versioning",
        "",
        "Adapters are versioned (v1, v2, ...). Each weekly iteration produces a",
        "new version. The evaluation harness tracks accuracy across versions to",
        "ensure improvements are monotonic and no regression occurs.",
        "",
        "## Langfuse Integration",
        "",
        "Every LLM call trace in Langfuse includes the model version tag",
        "(`base` vs `lora-v1`) so reviewers can trace which adapter was active",
        "for any given classification decision.",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Package LoRA adapter for LM Studio"
    )
    parser.add_argument(
        "--adapter-source", type=Path, required=True,
        help="Directory containing the trained adapter weights",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/adapters/packaged"),
    )
    parser.add_argument(
        "--name", default="qwen3.6-35b-a3b-shieldpoint-v1",
        help="Name for the packaged adapter",
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--metrics", type=Path, default=None,
        help="Path to metrics JSON from evaluate_adapter.py",
    )
    parser.add_argument(
        "--training-summary", type=Path, default=None,
        help="Path to training summary JSON from lora_train.py",
    )
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    metrics = None
    if args.metrics and args.metrics.exists():
        with open(args.metrics) as f:
            metrics = json.load(f)

    training_summary = None
    if args.training_summary and args.training_summary.exists():
        with open(args.training_summary) as f:
            training_summary = json.load(f)

    package_dir = package_adapter(
        adapter_source=args.adapter_source,
        output_dir=args.output_dir,
        adapter_name=args.name,
        base_model=args.base_model,
        metrics=metrics,
        training_summary=training_summary,
        create_zip=not args.no_zip,
    )
    print(f"Packaged adapter: {package_dir}")


if __name__ == "__main__":
    main()
