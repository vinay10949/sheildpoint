# ShieldPoint Claim Intake Automation (SP-203)

Automated claim intake pipeline that replaces the manual 2.3-day process for
claims arriving via web portal, email (IMAP polling), and fax (digitised
PDF). Uses Tesseract OCR to extract structured data from fax/email
attachments, validates the extracted data against required fields, and
formats the claim into the standard JSON schema expected by the
IntakeAgent downstream.

## Architecture

```
                    ┌──────────────────┐
                    │   Web Portal UI  │
                    └────────┬─────────┘
                             │ POST /intake/claims
                             ▼
┌────────────┐    ┌───────────────────────────────┐
│  IMAP      │───▶│                               │
│  Mailbox   │    │     Claim Intake Service      │
└────────────┘    │  (FastAPI + background poller)│
                  │                               │     ┌──────────────┐
┌────────────┐    │   ┌──────────────────────┐    │────▶│ Accepted     │
│  Fax       │───▶│   │  OCR (Tesseract)     │    │     │ Claims Store │──▶ IntakeAgent
│  Gateway   │    │   │  + field extractor   │    │     └──────────────┘
└────────────┘    │   │  + field validator   │    │     ┌──────────────┐
                  │   └──────────────────────┘    │────▶│ Manual       │
                  │                               │     │ Review Queue │
                  └───────────────────────────────┘     └──────────────┘
```

| Source | Endpoint | Pipeline | Latency Target |
|--------|----------|----------|----------------|
| Web portal | `POST /intake/claims` | direct validation (no OCR) | < 30s |
| Email | `POST /intake/claims/email` (manual) or IMAP poller (auto) | OCR + extraction + validation | < 2 min |
| Fax | `POST /intake/claims/fax` (multipart PDF) | OCR + extraction + validation | < 2 min |

## Standard Claim JSON

The pipeline normalises every accepted claim into this schema, which is
the contract with the downstream IntakeAgent:

```json
{
  "policyholder_name": "Alice Homeowner",      // required
  "policy_id": "HO-2024-001",                   // required
  "claim_type": "homeowners",                   // required (enum)
  "date_of_loss": "2026-03-14",                 // required (ISO-8601)
  "damage_description": "Wind damage to roof.", // required
  "amount_claimed": 1250.00,                    // optional
  "incident_location": "123 Main St",           // optional
  "adjuster_id": null,                          // optional
  "phone": "(555) 123-4567",                    // optional
  "email": "alice@example.com"                  // optional
}
```

## Quick Start

### Install

```bash
cd claim_intake/
pip install -e ".[dev]"
```

System dependencies (must be on PATH):
- `tesseract-ocr` — for OCR
- `poppler-utils` — for `pdftoppm` (PDF rasterisation)

### Run the API server

```bash
python -m claim_intake.api
# → starts on http://localhost:8001
```

### Submit a web claim

```bash
curl -X POST http://localhost:8001/intake/claims \
  -H 'Content-Type: application/json' \
  -d '{
    "policyholder_name": "Alice Homeowner",
    "policy_id": "HO-2024-001",
    "claim_type": "homeowners",
    "date_of_loss": "2026-03-14",
    "damage_description": "Wind damage to roof shingles."
  }'
```

Response:

```json
{
  "claim_id": "CLM-2026-ABCD1234EF",
  "status": "accepted",
  "source": "web",
  "accepted": true,
  "claim": { ... },
  "errors": [],
  "latency_sec": 0.003,
  "request_id": "REQ-..."
}
```

### Submit a fax PDF

```bash
curl -X POST http://localhost:8001/intake/claims/fax \
  -F 'file=@fax.pdf' \
  -F 'fax_number=+1-555-0100'
```

### Configure IMAP polling

Set these env vars and restart the server — the poller auto-starts on
startup and polls every 60 seconds:

```bash
export INTAKE_IMAP_ENABLED=true
export INTAKE_IMAP_HOST=imap.gmail.com
export INTAKE_IMAP_USER=claims@shieldpoint.example
export INTAKE_IMAP_PASSWORD='app-specific-token'
export INTAKE_IMAP_MAILBOX=INBOX
export INTAKE_IMAP_POLL_INTERVAL=60
```

### Manual review queue

```bash
# List pending review items
curl http://localhost:8001/review/queue

# Get a specific item
curl http://localhost:8001/review/queue/CLM-2026-XXXX

# Pop the next item (FIFO)
curl -X POST http://localhost:8001/review/queue/next

# Resolve: accept (with corrected fields) or reject
curl -X POST http://localhost:8001/review/queue/CLM-2026-XXXX/resolve \
  -H 'Content-Type: application/json' \
  -d '{
    "decision": "accept",
    "claim": {
      "policyholder_name": "Bob Driver",
      "policy_id": "AU-2024-015",
      "claim_type": "auto",
      "date_of_loss": "2026-04-02",
      "damage_description": "Rear-end collision on highway."
    }
  }'
```

## Load Test

Run 100 concurrent claims against an in-process API server:

```bash
python scripts/run_load_test.py --count 100 --concurrency 100 --in-process
```

Sample output (on a typical dev machine):

```
======================================================================
SP-203 Load Test Results
======================================================================
  Claims submitted:   100
  Concurrency:        100
  Total elapsed:      0.18s
  Throughput:         565.8 claims/sec

Latency (seconds):
  min:   0.122
  mean:  0.158
  P50:   0.163
  P95:   0.169
  P99:   0.171  (AC: < 30.000)
  max:   0.171

Outcomes:
  accepted: 100
  in_review: 0
  rejected: 0
  http_errors: 0

======================================================================
PASS: P99 < 30s and all claims succeeded.
```

The P99 latency of **~0.17s** is **175x faster** than the 30s AC.

## Tests

```bash
pytest                                  # full suite (106 tests)
pytest -m "not slow"                    # exclude OCR + load tests
pytest -m load                          # only load tests
pytest --cov=claim_intake               # with coverage report
```

Test coverage: 83% overall, 96%+ on validator/store/schemas/config.

## Module Layout

```
claim_intake/
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/claim_intake/
│   ├── __init__.py          # public API
│   ├── config.py            # IntakeConfig (env-driven)
│   ├── schemas.py           # Pydantic models + StandardClaim
│   ├── store.py             # in-memory stores (accepted + review queue)
│   ├── ocr.py               # Tesseract wrapper
│   ├── extractor.py         # regex + LLM-assisted field extraction
│   ├── validator.py         # required/optional field validation
│   ├── pipeline.py          # end-to-end orchestration
│   ├── email_poller.py      # background IMAP poller
│   └── api.py               # FastAPI app
├── tests/
│   ├── conftest.py
│   ├── test_ocr.py          # includes >95% accuracy AC test
│   ├── test_extractor.py
│   ├── test_validator.py
│   ├── test_pipeline.py     # end-to-end web/email/fax
│   ├── test_api.py          # FastAPI endpoint tests
│   ├── test_email_poller.py # mocked IMAP server
│   └── test_load.py         # 100 concurrent claims, P99 < 30s
├── scripts/
│   └── run_load_test.py     # standalone load test CLI
└── samples/                 # sample fax PDFs for manual testing
```

## Acceptance Criteria Mapping

| AC | Where |
|----|-------|
| Web portal API endpoint accepts claim submissions and returns claim ID | `POST /intake/claims` in `api.py` |
| Email polling retrieves claim emails every 60 seconds, extracts attachments, runs OCR | `EmailPoller` in `email_poller.py` (default interval 60s) |
| Fax PDF ingestion with OCR achieves > 95% character accuracy on typed text | `test_ocr.py::TestTesseractPath::test_typed_text_accuracy` |
| Structured data extraction maps OCR output to standard claim JSON schema | `extractor.py` + `schemas.StandardClaim` |
| Field validation catches missing/invalid fields with specific error messages | `validator.py` + `FieldError` in `schemas.py` |
| Invalid claims routed to manual review queue with error details | `store.put_review` + `ReviewItem` |
| End-to-end intake latency < 30 seconds for web claims | `test_pipeline.py::test_web_claim_latency_under_30s` + load test |
| End-to-end intake latency < 2 minutes for OCR claims | `test_pipeline.py::test_fax_latency_under_2min` |
| Load test: 100 concurrent claims via API, measure P99 latency | `test_load.py` + `scripts/run_load_test.py` |

## Configuration Reference

All env vars (read at call time via `IntakeConfig.from_env()`):

| Variable | Default | Description |
|----------|---------|-------------|
| `INTAKE_API_HOST` | `0.0.0.0` | API bind host |
| `INTAKE_API_PORT` | `8001` | API bind port (avoids clash with agent API on 8000) |
| `INTAKE_WEB_CLAIM_TIMEOUT_SEC` | `30` | Web-claim latency budget |
| `OCR_CLAIM_TIMEOUT_SEC` | `120` | OCR-claim latency budget |
| `TESSERACT_CMD` | `tesseract` | Path to tesseract binary |
| `OCR_DPI` | `300` | DPI for PDF rasterisation |
| `OCR_LANG` | `eng` | Tesseract language |
| `OCR_MAX_PAGES` | `25` | Cap on pages per fax |
| `INTAKE_IMAP_ENABLED` | `false` | Enable IMAP poller |
| `INTAKE_IMAP_HOST` | — | IMAP server hostname |
| `INTAKE_IMAP_PORT` | `993` | IMAP server port |
| `INTAKE_IMAP_SSL` | `true` | Use SSL |
| `INTAKE_IMAP_USER` | — | Mailbox login |
| `INTAKE_IMAP_PASSWORD` | — | Mailbox password |
| `INTAKE_IMAP_MAILBOX` | `INBOX` | Mailbox to poll |
| `INTAKE_IMAP_POLL_INTERVAL` | `60` | Seconds between polls |
| `INTAKE_IMAP_SEARCH_SINCE_DAYS` | `7` | Only poll recent mail |
| `REVIEW_MAX_MISSING_FIELDS` | `5` | Reject if >= N required fields missing |
| `LM_STUDIO_BASE_URL` | — | LLM endpoint for assisted extraction (optional) |
| `LM_STUDIO_API_KEY` | `lm-studio` | LLM API key |
| `QWEN_MODEL_ID` | `qwen3.6-35b-a3b-q4_k_m` | LLM model ID |

## Integration with IntakeAgent

The IntakeAgent (downstream) reads accepted claims from the in-memory
store via `claim_intake.store.get_accepted(claim_id)`. In production this
would be replaced with a Postgres-backed store, but the API surface
stays the same.

To run end-to-end (intake + agent):

```python
from claim_intake import intake_web_claim, WebClaimSubmission
from claim_intake.store import get_accepted
from shieldpoint_agents import Agent

# 1. Intake
result = intake_web_claim(WebClaimSubmission(...))
if result.accepted:
    # 2. Hand off to IntakeAgent
    claim_dict = get_accepted(result.claim_id)
    agent = Agent(...)
    decision = agent.run(claim_dict["claim"])
```
