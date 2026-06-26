# ShieldPoint Agent Evaluation Report

Generated: `2026-06-24T13:37:31.473026+00:00`  
Total claims evaluated: **750**

## Headline Metrics

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| Decision accuracy | 0.8907 | 0.88 | ▲ +0.0107 |
| Severity accuracy | 0.8533 | 0.85 | ▲ +0.0033 |
| Claim-type accuracy | 0.9293 | 0.92 | ▲ +0.0093 |
| Fraud F1 | 0.764 | 0.8135 | ▼ -0.0495 |
| Payout MAE (USD) | 224.37 | 850.0 | ▼ -625.6300 |
| Payout within ±10% | 0.468 | 0.62 | ▼ -0.1520 |
| Days-to-settle MAE | 2.16 | 4.5 | ▼ -2.3400 |

## Decision Confusion Matrix

_Rows = ground truth, columns = predicted._

| | approve | deny | route_to_manual_review |
|---|---|---|---|
| approve | 353 | 23 | 21 |
| deny | 7 | 124 | 7 |
| route_to_manual_review | 15 | 9 | 191 |

## Per-Class Decision Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| approve | 0.9413 | 0.8892 | 0.9145 | 397 |
| deny | 0.7949 | 0.8986 | 0.8435 | 138 |
| route_to_manual_review | 0.8721 | 0.8884 | 0.8802 | 215 |

## Fraud Detection

- Precision: **0.7556**
- Recall: **0.7727**
- F1: **0.7640**
- TP/FP/FN/TN: 34/11/10/695

## Payout Accuracy

- MAE: **$224.37**
- RMSE: **$337.45**
- Within ±10% of gold: **46.8%**
- Within ±25% of gold: **54.7%**

## Source Distribution

- `fallback`: 34 (4.5%)
- `hitl_escalation`: 80 (10.7%)
- `llm`: 636 (84.8%)

## Confidence Stats

- Mean: **0.7520**
- Min: **0.5509**
- Max: **0.9496**
- Count: 750

## Regression Check

- Threshold: -0.0500 (delta below baseline)
- **Result: PASS**
