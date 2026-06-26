# ShieldPoint Evaluation Dataset & Baseline Metrics Bundle

Bundle for the "Create Evaluation Dataset and Baseline Metrics" Jira ticket.

## Layout

```
scripts/
  generate_dataset.py          # 5,000-claim dataset generator (seed=42)
  generate_baseline_metrics.py # Baseline metrics w/ Monte Carlo variance bands

download/
  dataset/
    train.csv  train.jsonl     # 3,500 claims (LoRA fine-tuning, Epic 2)
    val.csv    val.jsonl       # 750 claims (hyperparameter tuning)
    test.csv   test.jsonl      # 750 claims (final accuracy + Langfuse eval)
    dataset_manifest.json      # Counts, schema, per-split SHA-256
    DATASET_README.md          # Schema + distribution docs
  langfuse/
    upload_to_langfuse.py      # Pushes test set as Langfuse datasetitems
    dry_run_sample.json        # 3-item payload preview
    dry_run_full.jsonl         # All 750 items in Langfuse payload format
  baseline_metrics/
    baseline_metrics.json      # Manual-adjuster baseline + MC variance + per-class
  eval/
    automated_eval.py          # Compares agent predictions vs test set
    demo_predictions.jsonl     # Sample predictions at baseline accuracy (demo)
    eval_report.json           # Full metrics bundle (machine-readable)
    eval_report.junit.xml      # JUnit XML for CI dashboards
    eval_report.md             # Human-readable Markdown summary
```

## Quick Start

```bash
# 1. Regenerate the dataset (deterministic — same seed, same output)
python3 scripts/generate_dataset.py

# 2. Regenerate baseline metrics (with 200-run Monte Carlo)
python3 scripts/generate_baseline_metrics.py

# 3. Preview the Langfuse payload without uploading
python3 download/langfuse/upload_to_langfuse.py --dry-run --limit 3

# 4. Push the test set to a live Langfuse instance
python3 download/langfuse/upload_to_langfuse.py \
    --host http://localhost:3000 \
    --public-key pk-lf-... \
    --secret-key sk-lf-...

# 5. Run the automated evaluation against agent predictions
python3 download/eval/automated_eval.py \
    --predictions /path/to/agent_predictions.jsonl \
    --baseline download/baseline_metrics/baseline_metrics.json \
    --regression-threshold -0.02
# Exit 0 = PASS, Exit 1 = REGRESSION, Exit 2 = ERROR

# 6. Demo run (no real agent needed — generates synthetic predictions)
python3 download/eval/automated_eval.py --demo
```

## Prediction JSONL Format

Each line of the predictions file must be:

```json
{
  "claim_id": "CLM-2026-XXXXX",
  "decision": "approve|deny|route_to_manual_review",
  "severity": "low|medium|high|catastrophic",
  "claim_type": "homeowners|auto|health",
  "fraud_flag": true,
  "payout_amount": 3824.98,
  "days_to_settle": 30,
  "confidence": 0.85,
  "source": "llm|fallback|hitl_escalation"
}
```

## Acceptance Criteria Coverage

| AC | Status | Artifact |
|----|--------|----------|
| 5,000 historical claims labeled | DONE | dataset/ (train+val+test = 5,000) |
| Split 3,500 / 750 / 750 | DONE | dataset_manifest.json |
| PII anonymized, financial data preserved | DONE | SHA-1-based synthetic IDs; financial fields untouched |
| Langfuse evaluation dataset for test set | DONE | langfuse/upload_to_langfuse.py + dry_run_full.jsonl |
| Baseline metrics per classification type | DONE | baseline_metrics/baseline_metrics.json (per_class block) |
| Automated eval script after each deployment | DONE | eval/automated_eval.py (JSON + JUnit XML + Markdown + exit code) |

## Key Design Decisions

- **Deterministic generation** (seed=42): re-running `generate_dataset.py`
  produces an identical dataset, so regression comparisons are stable.
- **Anonymization**: SHA-1(raw_claimant_name) -> `CLAIMANT-XXXXXX` token.
  Same person always maps to the same token across splits and reloads.
  Financial fields preserved verbatim per AC.
- **Line mix**: 50% homeowners / 30% auto / 20% health (matches
  ShieldPoint's primary book).
- **Baseline rates** are industry-typical for manual P&C/A&H adjustment
  (LexisNexis 2023 disposition, McKinsey 2024 severity, Coalition Against
  Insurance Fraud 2024 for fraud). Each metric includes a 95% Monte Carlo
  confidence band from 200 simulated adjuster runs.
- **CI/CD ready**: the eval script emits JSON + JUnit XML + Markdown and
  exits non-zero on regression — drop-in for GitHub Actions / Jenkins.
