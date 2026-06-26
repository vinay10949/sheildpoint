"""Unit tests for the IMAP email poller — uses a mock IMAP server."""

from __future__ import annotations

import email
import email.utils
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from claim_intake.config import IntakeConfig
from claim_intake.email_poller import EmailPoller, PollResult, _parse_message
from claim_intake.schemas import EmailClaimSubmission, IntakeResult


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------
class TestParseMessage:
    def test_extracts_sender_and_date(self):
        msg = email.message_from_string(
            "From: adjuster@example.com\r\n"
            "Date: Sat, 15 Mar 2026 10:00:00 +0000\r\n"
            "Subject: Test\r\n"
            "\r\n"
            "Body text"
        )
        sender, received_at, attachments = _parse_message(msg.as_bytes())
        assert sender == "adjuster@example.com"
        assert "2026-03-15" in received_at
        assert attachments == []  # no attachments in this message

    def test_extracts_pdf_attachment(self, make_fax_pdf):
        pdf_bytes = make_fax_pdf()
        # Build a multipart message with a PDF attachment.
        msg = email.message.EmailMessage()
        msg["From"] = "adjuster@example.com"
        msg["Date"] = email.utils.formatdate(localtime=False)
        msg["Subject"] = "Claim form"
        msg.set_content("Please find attached claim form.")
        msg.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename="claim.pdf",
        )
        sender, received_at, attachments = _parse_message(msg.as_bytes())
        assert sender == "adjuster@example.com"
        assert len(attachments) == 1
        filename, mime, data = attachments[0]
        assert filename == "claim.pdf"
        assert mime == "application/pdf"
        assert data == pdf_bytes

    def test_extracts_inline_image_as_attachment(self):
        # An inline image (no Content-Disposition: attachment) with an
        # image/* MIME type should still be picked up.
        msg = email.message.EmailMessage()
        msg["From"] = "user@example.com"
        msg["Subject"] = "Inline screenshot"
        msg.set_content("See attached.")
        # Tiny 1x1 PNG.
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c63000100000005000100"
            "0d0a2db4000000004945"
            "4e44ae426082"
        )
        msg.add_attachment(
            png_bytes, maintype="image", subtype="png", filename="damage.png",
        )
        sender, received_at, attachments = _parse_message(msg.as_bytes())
        assert len(attachments) == 1
        assert attachments[0][0] == "damage.png"

    def test_malformed_date_falls_back_to_now(self):
        msg = email.message_from_string(
            "From: adjuster@example.com\r\n"
            "Date: not-a-date\r\n"
            "Subject: Test\r\n"
            "\r\n"
            "Body"
        )
        sender, received_at, attachments = _parse_message(msg.as_bytes())
        assert sender == "adjuster@example.com"
        # Should have a valid ISO timestamp even with bad input.
        datetime.fromisoformat(received_at)


# ---------------------------------------------------------------------------
# Poller with mocked IMAP connection
# ---------------------------------------------------------------------------
class TestPollOnceMocked:
    """Mock the IMAP connection so we don't need a real server."""

    def _build_mock_conn(self, messages: list[tuple[str, bytes]]) -> Any:
        """Build a mock imaplib.IMAP4 object.

        ``messages`` is a list of (uid, raw_bytes) tuples.
        """
        conn = MagicMock()
        # conn.select returns ("OK", [b"3"])
        conn.select.return_value = ("OK", [str(len(messages)).encode()])
        # conn.search returns the list of UIDs.
        uids = [uid.encode() for uid, _ in messages]
        conn.search.return_value = ("OK", [b" ".join(uids)] if uids else [b""])
        # conn.fetch returns the message bytes per UID.
        def _fetch(uid_bytes, fmt):
            uid = uid_bytes.decode() if isinstance(uid_bytes, bytes) else uid_bytes
            for u, raw in messages:
                if u == uid:
                    return ("OK", [(uid_bytes, raw)])
            return ("OK", [None])
        conn.fetch.side_effect = _fetch
        conn.logout.return_value = None
        return conn

    def test_poll_once_with_no_messages(self, test_config):
        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=True, imap_host="mock"),
        )
        mock_conn = self._build_mock_conn([])
        with patch("claim_intake.email_poller._connect", return_value=mock_conn):
            result = poller.poll_once()
        assert result.messages_found == 0
        assert result.messages_processed == 0
        assert result.errors == 0

    def test_poll_once_with_unprocessed_messages(
        self, test_config, make_fax_pdf
    ):
        """A message with a PDF attachment should be fed to the processor."""
        pdf_bytes = make_fax_pdf()
        msg = email.message.EmailMessage()
        msg["From"] = "adjuster@example.com"
        msg["Date"] = email.utils.formatdate(localtime=False)
        msg["Subject"] = "Claim form"
        msg.set_content("Please find attached claim form.")
        msg.add_attachment(
            pdf_bytes, maintype="application", subtype="pdf",
            filename="claim.pdf",
        )

        processed: list[EmailClaimSubmission] = []
        def processor(sub: EmailClaimSubmission) -> IntakeResult:
            processed.append(sub)
            # Return a stub IntakeResult so the poller doesn't blow up.
            from claim_intake.schemas import (
                ClaimStatus, IntakeResult, IntakeSource, new_claim_id,
            )
            return IntakeResult(
                claim_id=new_claim_id(),
                status=ClaimStatus.ACCEPTED,
                source=IntakeSource.EMAIL,
                accepted=True,
            )

        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=True, imap_host="mock"),
            processor=processor,
        )
        mock_conn = self._build_mock_conn([("100", msg.as_bytes())])
        with patch("claim_intake.email_poller._connect", return_value=mock_conn):
            result = poller.poll_once()
        assert result.messages_found == 1
        assert result.messages_processed == 1
        assert result.errors == 0
        assert len(processed) == 1
        assert processed[0].sender == "adjuster@example.com"
        assert len(processed[0].attachments) == 1

    def test_poll_once_skips_messages_without_attachments(self):
        """A message with no attachments should be marked processed but not
        fed to the processor."""
        msg = email.message_from_string(
            "From: user@example.com\r\n"
            "Date: Sat, 15 Mar 2026 10:00:00 +0000\r\n"
            "Subject: No attachments\r\n"
            "\r\n"
            "Just a text email"
        )
        processed: list[EmailClaimSubmission] = []
        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=True, imap_host="mock"),
            processor=lambda s: processed.append(s) or _stub_result(),
        )
        mock_conn = self._build_mock_conn([("100", msg.as_bytes())])
        with patch("claim_intake.email_poller._connect", return_value=mock_conn):
            result = poller.poll_once()
        assert result.messages_found == 1
        assert result.messages_processed == 0  # skipped
        assert len(processed) == 0

    def test_poll_once_does_not_reprocess_already_processed(self, make_fax_pdf):
        """Second poll cycle should not re-process the same UID."""
        pdf_bytes = make_fax_pdf()
        msg = email.message.EmailMessage()
        msg["From"] = "adjuster@example.com"
        msg["Date"] = email.utils.formatdate(localtime=False)
        msg["Subject"] = "Claim"
        msg.set_content("Body")
        msg.add_attachment(
            pdf_bytes, maintype="application", subtype="pdf",
            filename="claim.pdf",
        )

        processed_count = [0]
        def processor(sub):
            processed_count[0] += 1
            return _stub_result()

        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=True, imap_host="mock"),
            processor=processor,
        )
        mock_conn = self._build_mock_conn([("100", msg.as_bytes())])
        with patch("claim_intake.email_poller._connect", return_value=mock_conn):
            r1 = poller.poll_once()
            r2 = poller.poll_once()
        assert r1.messages_processed == 1
        # Second poll — UID 100 is already in _processed_uids, so it's
        # filtered out before fetching.
        assert r2.messages_found == 0
        assert r2.messages_processed == 0
        assert processed_count[0] == 1  # only processed once

    def test_poll_once_handles_processor_exception(self, make_fax_pdf):
        """If the processor raises, the poller logs + counts the error and
        continues to the next message."""
        pdf_bytes = make_fax_pdf()
        msgs = []
        for uid in ("100", "101"):
            m = email.message.EmailMessage()
            m["From"] = "adjuster@example.com"
            m["Date"] = email.utils.formatdate(localtime=False)
            m["Subject"] = "Claim"
            m.set_content("Body")
            m.add_attachment(
                pdf_bytes, maintype="application", subtype="pdf",
                filename=f"claim_{uid}.pdf",
            )
            msgs.append((uid, m.as_bytes()))

        call_count = [0]
        def flaky_processor(sub):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First message fails")
            return _stub_result()

        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=True, imap_host="mock"),
            processor=flaky_processor,
        )
        mock_conn = self._build_mock_conn(msgs)
        with patch("claim_intake.email_poller._connect", return_value=mock_conn):
            result = poller.poll_once()
        assert result.messages_found == 2
        assert result.messages_processed == 1
        assert result.errors == 1
        assert result.first_error is not None

    def test_poll_once_handles_connect_failure(self):
        """If IMAP connect fails, poll_once returns an error result and
        does NOT raise."""
        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=True, imap_host="bad.example"),
        )
        with patch(
            "claim_intake.email_poller._connect",
            side_effect=RuntimeError("Connection refused"),
        ):
            result = poller.poll_once()
        assert result.first_error is not None
        assert "Connection refused" in result.first_error
        assert result.messages_found == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
class TestPollerLifecycle:
    def test_start_stop(self, test_config):
        poller = EmailPoller(
            config=IntakeConfig(imap_enabled=False),  # no real polling
        )
        assert not poller.is_running
        poller.start()
        assert poller.is_running
        poller.stop(timeout=2.0)
        assert not poller.is_running

    def test_double_start_is_idempotent(self, test_config):
        poller = EmailPoller(config=IntakeConfig(imap_enabled=False))
        poller.start()
        poller.start()  # should not start a second thread
        assert poller.is_running
        poller.stop(timeout=2.0)

    def test_stop_when_not_running_is_safe(self):
        poller = EmailPoller(config=IntakeConfig(imap_enabled=False))
        poller.stop(timeout=1.0)  # should not raise


def _stub_result() -> IntakeResult:
    """Build a stub IntakeResult for the mock processor."""
    from claim_intake.schemas import (
        ClaimStatus, IntakeResult, IntakeSource, new_claim_id,
    )
    return IntakeResult(
        claim_id=new_claim_id(),
        status=ClaimStatus.ACCEPTED,
        source=IntakeSource.EMAIL,
        accepted=True,
    )
