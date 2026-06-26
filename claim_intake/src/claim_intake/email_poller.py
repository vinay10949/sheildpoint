"""
IMAP email polling for claim intake (SP-203).

Polls a mailbox every ``config.imap_poll_interval_sec`` seconds, looks for
messages with attachments, and feeds each one through
:func:`pipeline.intake_email_claim`.

Design points:

- **Idempotent:** keeps an in-memory set of already-processed UIDs. In a
  real deployment this would be a persistent store (Redis or Postgres),
  but for the MVP the in-memory set is sufficient and avoids re-processing
  the same fax on every restart.
- **Graceful degradation:** if IMAP credentials are wrong or the server is
  unreachable, the poller logs the error and retries on the next tick.
  It does NOT crash the API server.
- **Background-safe:** the poller runs in a daemon thread so it doesn't
  block the FastAPI event loop. Concurrency is bounded — only one poll
  runs at a time, regardless of how long the previous one took.
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import logging
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .config import IntakeConfig
from .pipeline import intake_email_claim
from .schemas import EmailClaimSubmission, IntakeResult

logger = logging.getLogger("claim_intake.email_poller")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class PollResult:
    """The outcome of a single IMAP poll cycle."""

    #: Number of new messages found.
    messages_found: int = 0
    #: Number of messages successfully processed.
    messages_processed: int = 0
    #: Number of messages that errored during processing.
    errors: int = 0
    #: Total time spent on this poll cycle (seconds).
    elapsed_sec: float = 0.0
    #: The first error message, if any (for logging).
    first_error: str | None = None


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def _connect(config: IntakeConfig) -> imaplib.IMAP4 | imaplib.IMAP4_SSL:
    """Open an IMAP connection using the configured credentials."""
    if not config.imap_host:
        raise RuntimeError("IMAP host not configured (set INTAKE_IMAP_HOST)")

    if config.imap_ssl:
        ctx = ssl.create_default_context()
        conn = imaplib.IMAP4_SSL(config.imap_host, config.imap_port, ssl_context=ctx)
    else:
        conn = imaplib.IMAP4(config.imap_host, config.imap_port)

    conn.login(config.imap_user, config.imap_password)
    return conn


def _parse_message(raw: bytes) -> tuple[str, str, list[tuple[str, str, bytes]]]:
    """Parse a raw email message into (sender, received_at, attachments).

    Returns ISO-8601 timestamps for ``received_at``.
    """
    msg = email.message_from_bytes(raw)
    sender = msg.get("From", "").strip()
    date_hdr = msg.get("Date", "")
    try:
        dt = email.utils.parsedate_to_datetime(date_hdr)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            received_at = dt.isoformat()
        else:
            received_at = datetime.now(timezone.utc).isoformat()
    except (TypeError, ValueError):
        received_at = datetime.now(timezone.utc).isoformat()

    attachments: list[tuple[str, str, bytes]] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        disposition = part.get("Content-Disposition", "")
        if "attachment" not in disposition.lower():
            # Also accept inline PDFs/images as "attachments" for intake.
            content_type = part.get_content_type()
            if content_type not in ("application/pdf",) and not content_type.startswith("image/"):
                continue
        filename = part.get_filename() or "unnamed_attachment"
        mime_type = part.get_content_type() or "application/octet-stream"
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        attachments.append((filename, mime_type, payload))

    return sender, received_at, attachments


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------
class EmailPoller:
    """Background thread that polls IMAP for new claim emails.

    Usage::

        poller = EmailPoller(config=IntakeConfig.from_env())
        poller.start()   # starts the background thread
        # ... API server runs ...
        poller.stop()    # graceful shutdown

    Or, for testing, call :meth:`poll_once` directly to run a single cycle
    without starting the thread.
    """

    def __init__(
        self,
        *,
        config: IntakeConfig | None = None,
        processor: Callable[[EmailClaimSubmission], IntakeResult] | None = None,
    ):
        self._config = config or IntakeConfig.from_env()
        self._processor = processor or (lambda sub: intake_email_claim(sub, config=self._config))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._processed_uids: set[str] = set()
        self._last_poll: PollResult | None = None
        self._poll_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="claim-intake-email-poller", daemon=True,
        )
        self._thread.start()
        logger.info("Email poller started; interval=%ss", self._config.imap_poll_interval_sec)

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the poller to stop and wait briefly for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("Email poller stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_poll(self) -> PollResult | None:
        return self._last_poll

    @property
    def poll_count(self) -> int:
        return self._poll_count

    @property
    def processed_uid_count(self) -> int:
        return len(self._processed_uids)

    def reset_processed_uids(self) -> None:
        """Clear the in-memory set of processed UIDs (for tests)."""
        with self._lock:
            self._processed_uids.clear()

    # ------------------------------------------------------------------
    # Single poll cycle
    # ------------------------------------------------------------------
    def poll_once(self) -> PollResult:
        """Run one poll cycle synchronously and return the result.

        This is what the background thread calls on each tick. Tests can
        call it directly to avoid thread timing issues.
        """
        start = time.monotonic()
        result = PollResult()

        if not self._config.imap_enabled:
            logger.debug("IMAP disabled; skipping poll")
            result.elapsed_sec = time.monotonic() - start
            with self._lock:
                self._last_poll = result
            return result

        try:
            conn = _connect(self._config)
        except Exception as exc:
            logger.error("IMAP connect failed: %s", exc)
            result.first_error = str(exc)
            result.elapsed_sec = time.monotonic() - start
            with self._lock:
                self._last_poll = result
                self._poll_count += 1
            return result

        try:
            conn.select(self._config.imap_mailbox)
            # Search for messages received in the last N days.
            since_date = (datetime.now(timezone.utc) - timedelta(
                days=self._config.imap_search_since_days
            )).strftime("%d-%b-%Y")
            # IMAP SEARCH SINCE returns messages whose internal date is
            # on or after the given date.
            typ, data = conn.search(None, f'SINCE', since_date)
            if typ != "OK":
                logger.warning("IMAP SEARCH returned %s", typ)
                result.elapsed_sec = time.monotonic() - start
                with self._lock:
                    self._last_poll = result
                    self._poll_count += 1
                return result

            uids = [u.decode() for u in data[0].split()] if data and data[0] else []
            new_uids = [u for u in uids if u not in self._processed_uids]
            result.messages_found = len(new_uids)

            for uid in new_uids:
                try:
                    typ, msg_data = conn.fetch(uid.encode(), "(RFC822)")
                    if typ != "OK" or not msg_data or not msg_data[0]:
                        continue
                    raw = msg_data[0][1]
                    sender, received_at, attachments = _parse_message(raw)

                    if not attachments:
                        # No attachments — nothing to OCR. Mark processed
                        # so we don't keep re-fetching it.
                        with self._lock:
                            self._processed_uids.add(uid)
                        continue

                    submission = EmailClaimSubmission(
                        sender=sender,
                        received_at=received_at,
                        attachments=attachments,
                        subject=email.message_from_bytes(raw).get("Subject"),
                    )
                    self._processor(submission)
                    result.messages_processed += 1
                    with self._lock:
                        self._processed_uids.add(uid)
                except Exception as exc:
                    logger.exception("Failed to process UID %s: %s", uid, exc)
                    result.errors += 1
                    if result.first_error is None:
                        result.first_error = str(exc)
                    # Still mark as processed to avoid an infinite retry loop.
                    with self._lock:
                        self._processed_uids.add(uid)
        finally:
            try:
                conn.logout()
            except Exception:
                pass

        result.elapsed_sec = time.monotonic() - start
        with self._lock:
            self._last_poll = result
            self._poll_count += 1
        return result

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        interval = max(5, self._config.imap_poll_interval_sec)
        logger.info("Email poller loop started; interval=%ss", interval)
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                logger.exception("Unexpected poller error: %s", exc)
            # Wait for the interval, but check the stop event every second
            # so shutdown is responsive.
            for _ in range(interval):
                if self._stop_event.is_set():
                    return
                time.sleep(1)
