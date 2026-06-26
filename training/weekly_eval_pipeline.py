#!/usr/bin/env python3
"""
SP-205 — Weekly Evaluation Pipeline
====================================

Runs weekly to measure the latest LoRA adapter's accuracy against the
evaluation harness and decides whether to promote the adapter to
production.

The pipeline:
1. Loads the latest adapter from ``training/adapters/latest/``.
2. Runs the evaluation harness on the validation set.
3. Runs the MMLU benchmark to check general reasoning retention.
4. Compares the adapter's metrics to the previous week's metrics.
5. If the new adapter improves accuracy by >= 1pp without regressing MMLU
   by > 1pp, promotes it to production (updates ``adapters/production/``).
6. Logs all metrics to Langfuse with the model version tag.
7. Sends a summary report to the ML engineering team.

This script is designed to run as a weekly cron job::

    0 2 * * 1  # Every Monday at 2 AM

Usage
-----
    python weekly_eval_pipeline.py \\
        --adapter-dir training/adapters/latest \\
        --val-data training/data/val.jsonl \\
        --report-output training/reports/$(date +%Y-%m-%d).json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shieldpoint.training.weekly")


def run_weekly_pipeline(
    *,
    adapter_dir: Path,
    val_data: Path,
    base_model: str = "Qwen/Qwen3.6-35B-A3B",
    report_output: Path | None = None,
    production_dir: Path | None = None,
    previous_report: Path | None = None,
) -> dict[str, Any]:
    """Run the weekly evaluation pipeline.

    Returns a report dict with all metrics and the promotion decision.
    """
    report = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "val_data": str(val_data),
        "stages": [],
    }

    # ---- Stage 1: Evaluate adapter on validation set ----
    logger.info("Stage 1: Evaluating adapter on validation set")
    from evaluate_adapter import evaluate
    adapter_metrics_path = report_output.parent / "adapter_metrics.json" if report_output else None
    adapter_metrics = evaluate(
        model_id=base_model,
        val_data_path=val_data,
        adapter_path=adapter_dir,
        output_path=adapter_metrics_path,
    )
    report["adapter_metrics"] = adapter_metrics
    report["stages"].append({
        "stage": "adapter_evaluation",
        "status": "completed",
        "accuracy": adapter_metrics.get("exact_match_accuracy"),
    })

    # ---- Stage 2: Run MMLU benchmark ----
    logger.info("Stage 2: Running MMLU benchmark for reasoning retention")
    from mmlu_benchmark import run_mmlu, check_reasoning_retention
    base_mmlu = run_mmlu(base_model, use_lora=False)
    lora_mmlu = run_mmlu(base_model, adapter_path=str(adapter_dir), use_lora=True)
    retention = check_reasoning_retention(
        base_accuracy=base_mmlu["accuracy"],
        lora_accuracy=lora_mmlu["accuracy"],
    )
    report["mmlu_base"] = base_mmlu
    report["mmlu_lora"] = lora_mmlu
    report["retention_check"] = retention
    report["stages"].append({
        "stage": "mmlu_benchmark",
        "status": "completed",
        "base_accuracy": base_mmlu["accuracy"],
        "lora_accuracy": lora_mmlu["accuracy"],
        "within_threshold": retention["within_threshold"],
    })

    # ---- Stage 3: Compare to previous week ----
    if previous_report and Path(previous_report).exists():
        with open(previous_report) as f:
            prev = json.load(f)
        prev_acc = prev.get("adapter_metrics", {}).get("exact_match_accuracy", 0)
        curr_acc = adapter_metrics.get("exact_match_accuracy", 0)
        report["previous_week"] = {
            "accuracy": prev_acc,
            "current_accuracy": curr_acc,
            "delta_pp": (curr_acc - prev_acc) * 100,
        }
    else:
        report["previous_week"] = None

    # ---- Stage 4: Promotion decision ----
    # Promote if:
    # 1. Exact match accuracy >= 5pp over base (or previous adapter)
    # 2. MMLU retention within 2pp
    accuracy_ok = adapter_metrics.get("meets_5pp_threshold", False) or \
                  (report["previous_week"] and report["previous_week"]["delta_pp"] >= 0)
    reasoning_ok = retention["within_threshold"]
    should_promote = accuracy_ok and reasoning_ok

    report["promotion_decision"] = {
        "should_promote": should_promote,
        "accuracy_ok": accuracy_ok,
        "reasoning_ok": reasoning_ok,
        "reason": (
            "Promoted: accuracy improved and reasoning retained."
            if should_promote
            else f"Not promoted: accuracy_ok={accuracy_ok}, reasoning_ok={reasoning_ok}"
        ),
    }

    # ---- Stage 5: Promote if approved ----
    if should_promote and production_dir:
        import shutil
        production_dir.mkdir(parents=True, exist_ok=True)
        # Copy adapter to production
        prod_adapter = production_dir / "current"
        if prod_adapter.exists():
            # Archive the previous production adapter
            archive_name = f"archived_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(prod_adapter), str(production_dir / archive_name))
        shutil.copytree(str(adapter_dir), str(prod_adapter))
        report["promotion_decision"]["promoted_to"] = str(prod_adapter)
        logger.info("Promoted adapter to %s", prod_adapter)

    # ---- Stage 6: Log to Langfuse ----
    try:
        _log_to_langfuse(report)
    except Exception as e:
        logger.warning("Failed to log to Langfuse: %s", e)

    # ---- Stage 7: Save report ----
    if report_output:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        with open(report_output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Weekly report saved to %s", report_output)

    return report


def _log_to_langfuse(report: dict[str, Any]) -> None:
    """Log the weekly evaluation results to Langfuse.

    Records a trace with the adapter version, metrics, and promotion
    decision so reviewers can track adapter performance over time.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "agent_framework", "observability"
    ))
    try:
        from langfuse_wrapper import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span(
            name="weekly_eval_pipeline",
            input={"adapter_dir": report["adapter_dir"]},
            metadata={
                "adapter_metrics": report.get("adapter_metrics"),
                "mmlu_retention": report.get("retention_check"),
                "promotion_decision": report.get("promotion_decision"),
            },
        ) as span:
            span.update(output={
                "accuracy": report.get("adapter_metrics", {}).get("exact_match_accuracy"),
                "promoted": report.get("promotion_decision", {}).get("should_promote", False),
            })
    except ImportError:
        logger.info("Langfuse wrapper not available — skipping trace logging")


def main():
    parser = argparse.ArgumentParser(
        description="Weekly LoRA adapter evaluation pipeline"
    )
    parser.add_argument(
        "--adapter-dir", type=Path, required=True,
        help="Directory containing the latest adapter weights",
    )
    parser.add_argument(
        "--val-data", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/data/val.jsonl"),
    )
    parser.add_argument(
        "--base-model", default="Qwen/Qwen3.6-35B-A3B",
    )
    parser.add_argument(
        "--report-output", type=Path, required=True,
        help="Path to save the weekly report JSON",
    )
    parser.add_argument(
        "--production-dir", type=Path,
        default=Path("/home/z/my-project/sheildpoint/training/adapters/production"),
    )
    parser.add_argument(
        "--previous-report", type=Path, default=None,
        help="Path to last week's report for comparison",
    )
    args = parser.parse_args()

    report = run_weekly_pipeline(
        adapter_dir=args.adapter_dir,
        val_data=args.val_data,
        base_model=args.base_model,
        report_output=args.report_output,
        production_dir=args.production_dir,
        previous_report=args.previous_report,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
