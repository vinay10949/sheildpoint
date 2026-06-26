"""
Configuration for the Claim Intake Automation service (SP-203).

Reads from environment variables at call time (not at module-import time) so
tests that monkeypatch env vars work correctly. Mirrors the pattern already
used by ``shieldpoint_agents.config.AgentConfig``.

Environment variables
---------------------

Intake API
~~~~~~~~~~~

- ``INTAKE_API_HOST``           — default ``0.0.0.0``
- ``INTAKE_API_PORT``           — default ``8001``  (avoid clash with the
  existing agent API on 8000)
- ``INTAKE_WEB_CLAIM_TIMEOUT_SEC`` — default ``30`` (per AC: end-to-end web
  intake < 30s)

OCR pipeline
~~~~~~~~~~~~

- ``TESSERACT_CMD``             — path to tesseract binary; default ``tesseract``
- ``OCR_DPI``                   — default ``300`` (good for typed fax text)
- ``OCR_LANG``                  — default ``eng``
- ``OCR_MAX_PAGES``             — default ``25`` (cap on pages per fax)
- ``OCR_CLAIM_TIMEOUT_SEC``     — default ``120`` (per AC: < 2 min for OCR)

IMAP email polling
~~~~~~~~~~~~~~~~~~

- ``INTAKE_IMAP_ENABLED``       — default ``false`` (off by default; enable in prod)
- ``INTAKE_IMAP_HOST``          — IMAP server hostname (e.g. ``imap.gmail.com``)
- ``INTAKE_IMAP_PORT``          — default ``993``
- ``INTAKE_IMAP_SSL``           — default ``true``
- ``INTAKE_IMAP_USER``          — mailbox login
- ``INTAKE_IMAP_PASSWORD``      — mailbox password (use app-specific tokens)
- ``INTAKE_IMAP_MAILBOX``       — default ``INBOX``
- ``INTAKE_IMAP_POLL_INTERVAL`` — default ``60`` (per AC: every 60 seconds)
- ``INTAKE_IMAP_SEARCH_SINCE_DAYS`` — default ``7`` (only poll recent mail)

LLM-assisted parsing (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extractor falls back to a local LM Studio call when regex cannot parse a
field. If LM Studio is unreachable, regex-only is used and the claim is still
processed (with a lower confidence). These env vars are inherited from the
existing ``AgentConfig`` so we do not duplicate them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IntakeConfig:
    """Runtime configuration for the claim intake service.

    Construct via :meth:`from_env` in production, or directly in tests.
    """

    # ---- API server -------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    #: Hard ceiling for end-to-end web-claim intake (AC: < 30s).
    web_claim_timeout_sec: float = 30.0

    #: Hard ceiling for OCR-based claim intake (AC: < 2 min).
    ocr_claim_timeout_sec: float = 120.0

    # ---- OCR pipeline -----------------------------------------------------
    tesseract_cmd: str = "tesseract"
    ocr_dpi: int = 300
    ocr_lang: str = "eng"
    ocr_max_pages: int = 25

    # ---- IMAP email polling ----------------------------------------------
    imap_enabled: bool = False
    imap_host: str = ""
    imap_port: int = 993
    imap_ssl: bool = True
    imap_user: str = ""
    imap_password: str = ""
    imap_mailbox: str = "INBOX"
    imap_poll_interval_sec: int = 60
    imap_search_since_days: int = 7

    # ---- Manual review queue ---------------------------------------------
    #: If a single required field is missing/invalid, route to review.
    #: If more fields than this are missing, the claim is rejected outright
    #: (it is still persisted in the review queue with status="rejected").
    review_max_missing_fields: int = 5

    # ---- LLM-assisted extraction (optional) ------------------------------
    #: Base URL for an OpenAI-compatible LLM endpoint (e.g. LM Studio).
    #: If empty, the extractor falls back to regex-only.
    llm_base_url: str = ""
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen3.6-35b-a3b-q4_k_m"
    llm_timeout_sec: float = 10.0

    @classmethod
    def from_env(cls) -> "IntakeConfig":
        """Read configuration from environment variables at call time."""
        return cls(
            api_host=os.environ.get("INTAKE_API_HOST", "0.0.0.0"),
            api_port=int(os.environ.get("INTAKE_API_PORT", "8001")),
            web_claim_timeout_sec=float(
                os.environ.get("INTAKE_WEB_CLAIM_TIMEOUT_SEC", "30")
            ),
            ocr_claim_timeout_sec=float(
                os.environ.get("OCR_CLAIM_TIMEOUT_SEC", "120")
            ),
            tesseract_cmd=os.environ.get("TESSERACT_CMD", "tesseract"),
            ocr_dpi=int(os.environ.get("OCR_DPI", "300")),
            ocr_lang=os.environ.get("OCR_LANG", "eng"),
            ocr_max_pages=int(os.environ.get("OCR_MAX_PAGES", "25")),
            imap_enabled=os.environ.get("INTAKE_IMAP_ENABLED", "false").lower()
            in ("1", "true", "yes", "on"),
            imap_host=os.environ.get("INTAKE_IMAP_HOST", ""),
            imap_port=int(os.environ.get("INTAKE_IMAP_PORT", "993")),
            imap_ssl=os.environ.get("INTAKE_IMAP_SSL", "true").lower()
            in ("1", "true", "yes", "on"),
            imap_user=os.environ.get("INTAKE_IMAP_USER", ""),
            imap_password=os.environ.get("INTAKE_IMAP_PASSWORD", ""),
            imap_mailbox=os.environ.get("INTAKE_IMAP_MAILBOX", "INBOX"),
            imap_poll_interval_sec=int(
                os.environ.get("INTAKE_IMAP_POLL_INTERVAL", "60")
            ),
            imap_search_since_days=int(
                os.environ.get("INTAKE_IMAP_SEARCH_SINCE_DAYS", "7")
            ),
            review_max_missing_fields=int(
                os.environ.get("REVIEW_MAX_MISSING_FIELDS", "5")
            ),
            llm_base_url=os.environ.get("LM_STUDIO_BASE_URL", ""),
            llm_api_key=os.environ.get("LM_STUDIO_API_KEY", "lm-studio"),
            llm_model=os.environ.get("QWEN_MODEL_ID", "qwen3.6-35b-a3b-q4_k_m"),
            llm_timeout_sec=float(os.environ.get("LLM_TIMEOUT_SEC", "10")),
        )

    @property
    def llm_enabled(self) -> bool:
        """True iff an LLM base URL is configured."""
        return bool(self.llm_base_url)
