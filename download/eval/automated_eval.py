#!/usr/bin/env python3
"""
ShieldPoint Automated Evaluation Script
=======================================

Compares agent predictions against the 750-claim test set and produces a
machine-readable metrics bundle for CI/CD. Designed to run after every
deployment.

Inputs
------
  --ground-truth  Path to test.jsonl (default:
                  /home/z/my-project/download/dataset/test.jsonl)
  --predictions   Path to agent predictions JSONL. Each line must be:
                    {
                      "claim_id": "CLM-2026-XXXXX",
                      "decision": "approve" | "deny" | "route_to_manual_review",
                      "severity": "low" | "medium" | "high" | "catastrophic",
                      "claim_type": "homeowners" | "auto" | "health",
                      "fraud_flag": true | false,
                      "payout_amount": <float>,
                      "days_to_settle": <int>,        # optional
                      "confidence": <float>,           # optional
                      "source": "llm" | "fallback" | "hitl_escalation"
                    }
  --baseline      Path to baseline_metrics.json (default:
                  /home/z/my-project/download/baseline_metrics/baseline_metrics.json)
                  Used for regression detection.
  --regression-threshold   Accuracy delta below baseline that triggers a
                           non-zero exit code (default: -0.02 = -2pp).

Outputs (under --output-dir, default /home/z/my-project/download/eval/)
  eval_report.json         Full machine-readable metrics bundle
  eval_report.junit.xml    JUnit XML for CI dashboards (Jenkins, GH Actions)
  eval_report.md           Human-readable Markdown summary

Exit codes
----------
  0  All metrics within tolerance vs baseline (or no baseline provided)
  1  Regression detected (accuracy dropped > --regression-threshold below baseline)
  2  Hard error (file not found, malformed input, etc.)

Usage
-----
  # Compare agent predictions against the test set:
  python3 automated_eval.py \\
      --predictions /path/to/agent_predictions.jsonl

  # With a baseline to enforce regression gating:
  python3 automated_eval.py \\
      --predictions /path/to/agent_predictions.jsonl \\
      --baseline /home/z/my-project/download/baseline_metrics/baseline_metrics.json \\
      --regression-threshold -0.02

  # Demo run (uses a built-in baseline-as-predictions demo to produce a
  # sample report — useful for verifying the script end-to-end without a
  # real agent):
  python3 automated_eval.py --demo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_TEST_SET = Path("/home/z/my-project/download/dataset/test.jsonl")
DEFAULT_BASELINE = Path("/home/z/my-project/download/baseline_metrics/baseline_metrics.json")
DEFAULT_OUTPUT_DIR = Path("/home/z/my-project/download/eval")
DEFAULT_REGRESSION_THRESHOLD = -0.02  # -2pp accuracy drop


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
    return out


def join_pred_to_truth(truth: list[dict[str, Any]],
                       preds: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair predictions with their ground-truth claim records by claim_id."""
    truth_by_id = {t["claim_id"]: t for t in truth}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    missing = 0
    for p in preds:
        cid = p.get("claim_id")
        if cid not in truth_by_id:
            missing += 1
            continue
        pairs.append((truth_by_id[cid], p))
    if missing:
        print(f"[eval] WARN: {missing} predictions had no matching claim_id "
              f"in ground truth (skipped).", file=sys.stderr)
    return pairs


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------
def accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(preds)


def per_class_accuracy(preds: list[str], golds: list[str]) -> dict[str, dict[str, Any]]:
    """Per-class precision, recall, F1, support."""
    classes = sorted(set(golds) | set(preds))
    out: dict[str, dict[str, Any]] = {}
    for cls in classes:
        tp = sum(1 for p, g in zip(preds, golds) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(preds, golds) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(preds, golds) if p != cls and g == cls)
        support = sum(1 for g in golds if g == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        out[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return out


def confusion_matrix(preds: list[str], golds: list[str]) -> dict[str, dict[str, int]]:
    classes = sorted(set(golds) | set(preds))
    cm: dict[str, dict[str, int]] = {g: {p: 0 for p in classes} for g in classes}
    for p, g in zip(preds, golds):
        cm[g][p] += 1
    return cm


def mae(preds: list[float], golds: list[float]) -> float:
    if not preds:
        return 0.0
    return sum(abs(p - g) for p, g in zip(preds, golds)) / len(preds)


def rmse(preds: list[float], golds: list[float]) -> float:
    if not preds:
        return 0.0
    return math.sqrt(sum((p - g) ** 2 for p, g in zip(preds, golds)) / len(preds))


def within_tolerance(preds: list[float], golds: list[float],
                     tol_pct: float = 0.10) -> float:
    """Fraction of predictions within ±tol_pct of the gold value."""
    if not preds:
        return 0.0
    n = 0
    for p, g in zip(preds, golds):
        if g == 0:
            # If gold is 0, only count as correct if pred is also 0
            if p == 0:
                n += 1
            continue
        if abs(p - g) / g <= tol_pct:
            n += 1
    return n / len(preds)


def fraud_detection_metrics(pred_flags: list[bool], gold_flags: list[bool]) -> dict[str, Any]:
    """Precision/recall/F1 for the fraud_flag binary classifier."""
    tp = sum(1 for p, g in zip(pred_flags, gold_flags) if p and g)
    fp = sum(1 for p, g in zip(pred_flags, gold_flags) if p and not g)
    fn = sum(1 for p, g in zip(pred_flags, gold_flags) if not p and g)
    tn = sum(1 for p, g in zip(pred_flags, gold_flags) if not p and not g)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalReport:
    generated_at: str
    total_claims: int
    decision: dict[str, Any] = field(default_factory=dict)
    severity: dict[str, Any] = field(default_factory=dict)
    claim_type: dict[str, Any] = field(default_factory=dict)
    fraud: dict[str, Any] = field(default_factory=dict)
    payout: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    source_distribution: dict[str, int] = field(default_factory=dict)
    confidence_stats: dict[str, float] = field(default_factory=dict)
    regression: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def evaluate(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> EvalReport:
    """Compute all metrics from paired (truth, pred) records."""
    n = len(pairs)
    decisions_p = [p.get("decision", "unknown") for _, p in pairs]
    decisions_g = [t["expected_decision"] for t, _ in pairs]
    severity_p = [p.get("severity", "unknown") for _, p in pairs]
    severity_g = [t["expected_severity"] for t, _ in pairs]
    ctype_p = [p.get("claim_type", "unknown") for _, p in pairs]
    ctype_g = [t["expected_claim_type"] for t, _ in pairs]
    fraud_p = [bool(p.get("fraud_flag", False)) for _, p in pairs]
    fraud_g = [bool(t["expected_fraud_flag"]) for t, _ in pairs]
    payout_p = [float(p.get("payout_amount", 0.0)) for _, p in pairs]
    payout_g = [float(t["expected_payout_amount"]) for t, _ in pairs]
    settle_p = [int(p.get("days_to_settle", 0)) for _, p in pairs]
    settle_g = [int(t["days_to_settle"]) for t, _ in pairs]
    sources = [p.get("source", "unknown") for _, p in pairs]
    confidences = [float(p["confidence"]) for _, p in pairs if "confidence" in p]

    report = EvalReport(
        generated_at=datetime.now().astimezone().isoformat(),
        total_claims=n,
        decision={
            "accuracy": round(accuracy(decisions_p, decisions_g), 4),
            "per_class": per_class_accuracy(decisions_p, decisions_g),
            "confusion_matrix": confusion_matrix(decisions_p, decisions_g),
        },
        severity={
            "accuracy": round(accuracy(severity_p, severity_g), 4),
            "per_class": per_class_accuracy(severity_p, severity_g),
            "confusion_matrix": confusion_matrix(severity_p, severity_g),
        },
        claim_type={
            "accuracy": round(accuracy(ctype_p, ctype_g), 4),
            "per_class": per_class_accuracy(ctype_p, ctype_g),
        },
        fraud=fraud_detection_metrics(fraud_p, fraud_g),
        payout={
            "mae": round(mae(payout_p, payout_g), 2),
            "rmse": round(rmse(payout_p, payout_g), 2),
            "within_10pct": round(within_tolerance(payout_p, payout_g, 0.10), 4),
            "within_25pct": round(within_tolerance(payout_p, payout_g, 0.25), 4),
        },
        timeline={
            "days_to_settle_mae": round(mae([float(x) for x in settle_p],
                                            [float(x) for x in settle_g]), 2),
            "days_to_settle_rmse": round(rmse([float(x) for x in settle_p],
                                              [float(x) for x in settle_g]), 2),
        },
        source_distribution=dict(Counter(sources)),
        confidence_stats={
            "mean": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "min": round(min(confidences), 4) if confidences else 0.0,
            "max": round(max(confidences), 4) if confidences else 0.0,
            "count": len(confidences),
        },
    )
    return report


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------
def check_regression(report: EvalReport, baseline: dict[str, Any],
                     threshold: float) -> dict[str, Any]:
    """Compare key metrics against baseline. Returns regression dict."""
    metrics_to_check = [
        ("decision_accuracy",  report.decision["accuracy"],  baseline.get("decision_accuracy", 0.0)),
        ("severity_accuracy",  report.severity["accuracy"],  baseline.get("severity_accuracy", 0.0)),
        ("claim_type_accuracy", report.claim_type["accuracy"], baseline.get("claim_type_accuracy", 0.0)),
        ("fraud_f1",           report.fraud["f1"],            baseline.get("fraud_f1", 0.0)),
    ]
    regressions = []
    for name, actual, base in metrics_to_check:
        delta = actual - base
        if delta < threshold:
            regressions.append({
                "metric": name,
                "baseline": base,
                "actual": actual,
                "delta": round(delta, 4),
                "threshold": threshold,
            })
    return {
        "passed": len(regressions) == 0,
        "threshold": threshold,
        "regressions": regressions,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_json(report: EvalReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2))


def write_junit_xml(report: EvalReport, path: Path) -> None:
    """JUnit XML for CI dashboards. One testcase per metric category."""
    testsuites = ET.Element("testsuites", {
        "name": "shieldpoint-agent-eval",
        "tests": "5",
        "failures": "0" if report.regression.get("passed", True) else str(len(report.regression.get("regressions", []))),
    })
    ts = ET.SubElement(testsuites, "testsuite", {
        "name": "shieldpoint-agent-eval",
        "tests": "5",
        "failures": "0" if report.regression.get("passed", True) else str(len(report.regression.get("regressions", []))),
        "timestamp": report.generated_at,
    })

    cases = [
        ("decision_accuracy",  report.decision["accuracy"],  "decision"),
        ("severity_accuracy",  report.severity["accuracy"],  "severity"),
        ("claim_type_accuracy", report.claim_type["accuracy"], "claim_type"),
        ("fraud_f1",           report.fraud["f1"],           "fraud"),
        ("payout_within_10pct", report.payout["within_10pct"], "payout"),
    ]
    for name, value, classname in cases:
        tc = ET.SubElement(ts, "testcase", {
            "classname": classname,
            "name": name,
            "time": "0.0",
        })
        # Mark as failure if this metric regressed.
        for reg in report.regression.get("regressions", []):
            if reg["metric"] == name:
                ET.SubElement(tc, "failure", {
                    "message": f"{name}={value:.4f}, baseline={reg['baseline']:.4f}, "
                               f"delta={reg['delta']:.4f} < threshold {reg['threshold']}",
                    "type": "regression",
                })
                break

    # Add a system-out with the full report for log scraping.
    sysout = ET.SubElement(ts, "system-out")
    sysout.text = json.dumps(report.to_dict(), indent=2)

    tree = ET.ElementTree(testsuites)
    ET.indent(tree)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_markdown(report: EvalReport, path: Path,
                   baseline: dict[str, Any] | None) -> None:
    """Human-readable Markdown summary."""
    lines = [
        "# ShieldPoint Agent Evaluation Report",
        "",
        f"Generated: `{report.generated_at}`  ",
        f"Total claims evaluated: **{report.total_claims:,}**",
        "",
        "## Headline Metrics",
        "",
        "| Metric | Value | Baseline | Delta |",
        "|--------|-------|----------|-------|",
    ]
    metrics_table = [
        ("Decision accuracy",   report.decision["accuracy"],   baseline.get("decision_accuracy") if baseline else None),
        ("Severity accuracy",   report.severity["accuracy"],   baseline.get("severity_accuracy") if baseline else None),
        ("Claim-type accuracy", report.claim_type["accuracy"], baseline.get("claim_type_accuracy") if baseline else None),
        ("Fraud F1",            report.fraud["f1"],            baseline.get("fraud_f1") if baseline else None),
        ("Payout MAE (USD)",    report.payout["mae"],          baseline.get("payout_mae") if baseline else None),
        ("Payout within ±10%",  report.payout["within_10pct"], baseline.get("payout_within_10pct") if baseline else None),
        ("Days-to-settle MAE",  report.timeline["days_to_settle_mae"], baseline.get("days_to_settle_mae") if baseline else None),
    ]
    for name, val, base in metrics_table:
        if base is None:
            lines.append(f"| {name} | {val} | — | — |")
        else:
            delta = val - base
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            lines.append(f"| {name} | {val} | {base} | {arrow} {delta:+.4f} |")

    lines += [
        "",
        "## Decision Confusion Matrix",
        "",
        "_Rows = ground truth, columns = predicted._",
        "",
    ]
    cm = report.decision["confusion_matrix"]
    classes = sorted(cm.keys())
    lines.append("| | " + " | ".join(classes) + " |")
    lines.append("|---" * (len(classes) + 1) + "|")
    for g in classes:
        row = [g] + [str(cm[g][p]) for p in classes]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Per-Class Decision Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for cls, m in report.decision["per_class"].items():
        lines.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")

    lines += [
        "",
        "## Fraud Detection",
        "",
        f"- Precision: **{report.fraud['precision']:.4f}**",
        f"- Recall: **{report.fraud['recall']:.4f}**",
        f"- F1: **{report.fraud['f1']:.4f}**",
        f"- TP/FP/FN/TN: {report.fraud['tp']}/{report.fraud['fp']}/{report.fraud['fn']}/{report.fraud['tn']}",
        "",
        "## Payout Accuracy",
        "",
        f"- MAE: **${report.payout['mae']:,.2f}**",
        f"- RMSE: **${report.payout['rmse']:,.2f}**",
        f"- Within ±10% of gold: **{report.payout['within_10pct']:.1%}**",
        f"- Within ±25% of gold: **{report.payout['within_25pct']:.1%}**",
        "",
        "## Source Distribution",
        "",
    ]
    for src, n in sorted(report.source_distribution.items()):
        lines.append(f"- `{src}`: {n} ({n / report.total_claims:.1%})")

    if report.confidence_stats["count"] > 0:
        lines += [
            "",
            "## Confidence Stats",
            "",
            f"- Mean: **{report.confidence_stats['mean']:.4f}**",
            f"- Min: **{report.confidence_stats['min']:.4f}**",
            f"- Max: **{report.confidence_stats['max']:.4f}**",
            f"- Count: {report.confidence_stats['count']}",
        ]

    if report.regression:
        lines += [
            "",
            "## Regression Check",
            "",
            f"- Threshold: {report.regression['threshold']:+.4f} (delta below baseline)",
            f"- **Result: {'PASS' if report.regression['passed'] else 'FAIL'}**",
        ]
        if report.regression["regressions"]:
            lines += [
                "",
                "### Regressions Detected",
                "",
                "| Metric | Baseline | Actual | Delta |",
                "|--------|----------|--------|-------|",
            ]
            for reg in report.regression["regressions"]:
                lines.append(
                    f"| {reg['metric']} | {reg['baseline']:.4f} | {reg['actual']:.4f} | {reg['delta']:+.4f} |"
                )

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Demo mode — simulates predictions at the baseline accuracy level so the
# script can be tested end-to-end without a real agent.
# ---------------------------------------------------------------------------
def run_demo(test_set: Path, output_dir: Path) -> int:
    """Generate synthetic 'predictions' that match the baseline accuracy
    targets, then run the full eval pipeline. Useful for verifying the
    script and producing a sample report."""
    print(f"[eval] DEMO MODE -- generating synthetic predictions at baseline accuracy ...")
    truth = load_jsonl(test_set)
    rng = random.Random(123)

    # Per-AC baseline targets (industry-typical manual accuracy).
    baseline_targets = {
        "decision_accuracy":  0.88,  # 88% of decisions correct
        "severity_accuracy":  0.85,  # 85% of severity classifications correct
        "claim_type_accuracy": 0.92, # 92% of claim-type classifications correct
        "fraud_recall":       0.78,  # 78% of fraud cases caught
        "fraud_precision":    0.85,  # 85% of fraud flags correct
        "payout_mae":         850.0, # mean absolute error ~$850
    }

    preds: list[dict[str, Any]] = []
    for t in truth:
        # Decision: with prob = baseline accuracy, copy the gold; else flip.
        if rng.random() < baseline_targets["decision_accuracy"]:
            decision = t["expected_decision"]
        else:
            decision = rng.choice([d for d in ["approve", "deny", "route_to_manual_review"]
                                   if d != t["expected_decision"]])
        # Severity: same pattern.
        if rng.random() < baseline_targets["severity_accuracy"]:
            severity = t["expected_severity"]
        else:
            severity = rng.choice([s for s in ["low", "medium", "high", "catastrophic"]
                                   if s != t["expected_severity"]])
        # Claim type: easy to get right.
        if rng.random() < baseline_targets["claim_type_accuracy"]:
            ctype = t["expected_claim_type"]
        else:
            ctype = rng.choice([c for c in ["homeowners", "auto", "health"]
                                if c != t["expected_claim_type"]])
        # Fraud: model precision/recall independently.
        if t["expected_fraud_flag"]:
            fraud = rng.random() < baseline_targets["fraud_recall"]
        else:
            fraud = rng.random() < (1 - baseline_targets["fraud_precision"]) * 0.1
        # Payout: gold + gaussian noise.
        noise = rng.gauss(0, baseline_targets["payout_mae"] / 2)
        payout = max(0.0, round(t["expected_payout_amount"] + noise, 2))
        # Days to settle: gold ± 30%.
        settle = max(1, int(t["days_to_settle"] * rng.uniform(0.7, 1.3)))

        preds.append({
            "claim_id": t["claim_id"],
            "decision": decision,
            "severity": severity,
            "claim_type": ctype,
            "fraud_flag": fraud,
            "payout_amount": payout,
            "days_to_settle": settle,
            "confidence": round(rng.uniform(0.55, 0.95), 4),
            "source": rng.choices(["llm", "fallback", "hitl_escalation"],
                                  weights=[0.85, 0.05, 0.10])[0],
        })

    pred_path = output_dir / "demo_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"[eval] Wrote demo predictions -> {pred_path}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ground-truth", type=Path, default=DEFAULT_TEST_SET)
    p.add_argument("--predictions", type=Path,
                   help="Path to agent predictions JSONL")
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                   help="Path to baseline_metrics.json for regression gating")
    p.add_argument("--regression-threshold", type=float,
                   default=DEFAULT_REGRESSION_THRESHOLD,
                   help="Delta below baseline that triggers exit code 1 "
                        "(default: -0.02 = -2pp)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--demo", action="store_true",
                   help="Generate demo predictions at baseline accuracy "
                        "and exit. Useful for verifying the script.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        return run_demo(args.ground_truth, args.output_dir)

    if not args.predictions:
        # If no predictions given, default to demo mode for a sample report.
        print("[eval] No --predictions given; running in --demo mode for a "
              "sample report.", file=sys.stderr)
        run_demo(args.ground_truth, args.output_dir)
        args.predictions = args.output_dir / "demo_predictions.jsonl"

    # Load inputs.
    try:
        truth = load_jsonl(args.ground_truth)
        preds = load_jsonl(args.predictions)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    pairs = join_pred_to_truth(truth, preds)
    if not pairs:
        print("ERROR: no matching claim_ids between ground truth and predictions.",
              file=sys.stderr)
        return 2
    print(f"[eval] Paired {len(pairs)} predictions with ground truth.")

    # Evaluate.
    report = evaluate(pairs)

    # Regression check.
    baseline: dict[str, Any] | None = None
    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text())
        report.regression = check_regression(report, baseline,
                                             args.regression_threshold)
        if report.regression["passed"]:
            print("[eval] Regression check: PASS")
        else:
            print(f"[eval] Regression check: FAIL "
                  f"({len(report.regression['regressions'])} regressions)",
                  file=sys.stderr)
    else:
        print(f"[eval] No baseline at {args.baseline} -- skipping regression check.")

    # Write outputs.
    write_json(report, args.output_dir / "eval_report.json")
    write_junit_xml(report, args.output_dir / "eval_report.junit.xml")
    write_markdown(report, args.output_dir / "eval_report.md", baseline)

    print(f"[eval] Wrote:")
    print(f"[eval]   {args.output_dir / 'eval_report.json'}")
    print(f"[eval]   {args.output_dir / 'eval_report.junit.xml'}")
    print(f"[eval]   {args.output_dir / 'eval_report.md'}")

    # Exit code based on regression.
    if report.regression and not report.regression["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
