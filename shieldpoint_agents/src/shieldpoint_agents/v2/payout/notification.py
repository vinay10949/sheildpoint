"""
SP-405 — Claimant Notification Service
=======================================

Sends email notifications to claimants when their claim is paid out.
The notification includes:

- Payment confirmation (amount, ACH reference, settlement date)
- Link to download the PDF receipt
- Summary of the claim processing (agents involved, ZKP proofs)

In production, this uses a real email provider (e.g. SendGrid, AWS SES,
or an internal SMTP relay). In tests, the :class:`StubNotificationService`
logs the notification without sending a real email.
"""

from __future__ import annotations

import logging
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("shieldpoint.payout.notification")


@dataclass(frozen=True)
class NotificationResult:
    """Result of a notification send attempt.

    Attributes
    ----------
    success : bool
        True if the notification was sent (or stub-logged) successfully.
    recipient : str
        Email address of the recipient.
    message_id : str
        Unique message ID (for tracking).
    sent_at : float
        Unix timestamp.
    error : str, optional
        Error message if the send failed.
    """

    success: bool
    recipient: str
    message_id: str
    sent_at: float = field(default_factory=time.time)
    error: Optional[str] = None


@runtime_checkable
class NotificationService(Protocol):
    """Protocol for notification services."""

    def send_payment_confirmation(
        self,
        *,
        recipient_email: str,
        recipient_name: str,
        claim_id: str,
        payment_record: dict[str, Any],
        receipt_path: Optional[str] = None,
    ) -> NotificationResult: ...


class StubNotificationService:
    """Stub notification service for tests and local development.

    Logs the notification to the logger and stores it in an internal
    list for test assertions. Does NOT send a real email.
    """

    def __init__(self) -> None:
        self.sent_notifications: list[dict[str, Any]] = []

    def send_payment_confirmation(
        self,
        *,
        recipient_email: str,
        recipient_name: str,
        claim_id: str,
        payment_record: dict[str, Any],
        receipt_path: Optional[str] = None,
    ) -> NotificationResult:
        import uuid
        message_id = f"MSG-{uuid.uuid4().hex[:12].upper()}"

        notification = {
            "message_id": message_id,
            "recipient_email": recipient_email,
            "recipient_name": recipient_name,
            "claim_id": claim_id,
            "payment_id": payment_record.get("payment_id"),
            "amount": payment_record.get("net_payable"),
            "ach_reference": payment_record.get("ach_reference"),
            "receipt_path": receipt_path,
            "sent_at": time.time(),
        }
        self.sent_notifications.append(notification)

        logger.info(
            "STUB NOTIFICATION: Payment confirmation sent to %s for claim %s "
            "(amount=$%.2f, ACH ref=%s, message_id=%s)",
            recipient_email, claim_id,
            payment_record.get("net_payable", 0),
            payment_record.get("ach_reference", "N/A"),
            message_id,
        )

        return NotificationResult(
            success=True,
            recipient=recipient_email,
            message_id=message_id,
        )


class SMTPNotificationService:
    """SMTP-based notification service for production.

    Sends a real email with the PDF receipt attached.
    """

    def __init__(
        self,
        *,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: str = "claims@shieldpoint.example.com",
        from_name: str = "ShieldPoint Claims",
        use_tls: bool = True,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls

    def send_payment_confirmation(
        self,
        *,
        recipient_email: str,
        recipient_name: str,
        claim_id: str,
        payment_record: dict[str, Any],
        receipt_path: Optional[str] = None,
    ) -> NotificationResult:
        import uuid
        message_id = f"MSG-{uuid.uuid4().hex[:12].upper()}"

        msg = MIMEMultipart()
        msg["From"] = f"{self.from_name} <{self.from_email}>"
        msg["To"] = f"{recipient_name} <{recipient_email}>"
        msg["Subject"] = (
            f"ShieldPoint Payment Confirmation — Claim {claim_id} "
            f"(${payment_record.get('net_payable', 0):,.2f})"
        )
        msg["Message-ID"] = f"<{message_id}@shieldpoint.example.com>"

        # Build the email body
        body = self._build_email_body(
            recipient_name=recipient_name,
            claim_id=claim_id,
            payment_record=payment_record,
        )
        msg.attach(MIMEText(body, "html"))

        # Attach the PDF receipt if provided
        if receipt_path and Path(receipt_path).exists():
            with open(receipt_path, "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="pdf")
                attachment.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=Path(receipt_path).name,
                )
                msg.attach(attachment)

        # Send via SMTP
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            return NotificationResult(
                success=True,
                recipient=recipient_email,
                message_id=message_id,
            )
        except Exception as e:
            logger.error("Failed to send email to %s: %s", recipient_email, e)
            return NotificationResult(
                success=False,
                recipient=recipient_email,
                message_id=message_id,
                error=str(e),
            )

    def _build_email_body(
        self,
        *,
        recipient_name: str,
        claim_id: str,
        payment_record: dict[str, Any],
    ) -> str:
        """Build the HTML email body."""
        return f"""
<html><body style="font-family: Arial, sans-serif; color: #333;">
<h2>Payment Confirmation</h2>
<p>Dear {recipient_name},</p>
<p>Your insurance claim has been processed and payment has been initiated.
Here are the details:</p>
<table style="border-collapse: collapse; margin: 16px 0;">
<tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Claim ID:</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{claim_id}</td></tr>
<tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Payment Amount:</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">${payment_record.get('net_payable', 0):,.2f}</td></tr>
<tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>ACH Reference:</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{payment_record.get('ach_reference', 'N/A')}</td></tr>
<tr><td style="padding: 8px; border: 1px solid #ddd;"><strong>Expected Settlement:</strong></td>
<td style="padding: 8px; border: 1px solid #ddd;">{payment_record.get('settlement_date', '2-3 business days')}</td></tr>
</table>
<p>The detailed payment receipt is attached to this email as a PDF.
Please retain it for your records.</p>
<p>If you have any questions about this payment, please contact our claims
department at claims@shieldpoint.example.com or call 1-800-SHIELDPOINT.</p>
<p>Thank you for choosing ShieldPoint Insurance.</p>
<p style="color: #888; font-size: 12px; margin-top: 32px;">
This is an automated message. Please do not reply directly to this email.
</p>
</body></html>
"""
