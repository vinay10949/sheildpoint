"""
SP-405 — PDF Receipt Generator
===============================

Generates a professional PDF receipt for each claim payout, including:

- Claimant and policy information
- Payment breakdown (gross, deductible, co-pay, net payable)
- ACH reference and settlement date
- Full audit trail of all agents that processed the claim
- ZKP proof references (policy proof, compliance proof, fraud-detection proof)

The receipt is stored in Langfuse as part of the claim's audit trail
and emailed to the claimant as an attachment.

Implementation
--------------
Uses ReportLab for PDF generation (the same library used elsewhere in
the ShieldPoint stack for compliance documents). The receipt template
is intentionally simple and professional — no branding colors, just
clean typography suitable for printing.

For environments without ReportLab, falls back to a plain-text receipt
(.txt) so the payout pipeline still completes.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("shieldpoint.payout.receipt")

# Try to import ReportLab
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


@dataclass(frozen=True)
class ReceiptResult:
    """Result of receipt generation.

    Attributes
    ----------
    success : bool
        True if the receipt was generated successfully.
    file_path : str
        Path to the generated receipt file.
    file_format : str
        "pdf" or "txt" (fallback when ReportLab is unavailable).
    receipt_id : str
        Unique receipt ID (for tracking).
    generated_at : float
        Unix timestamp of generation.
    error : str, optional
        Error message if generation failed.
    """

    success: bool
    file_path: str
    file_format: str
    receipt_id: str
    generated_at: float = field(default_factory=time.time)
    error: Optional[str] = None


class ReceiptGenerator:
    """Generates PDF (or text) receipts for claim payouts.

    Parameters
    ----------
    output_dir : Path, optional
        Directory where receipts are saved. Defaults to ``/tmp/shieldpoint_receipts``.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("/tmp/shieldpoint_receipts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        payment_record: dict[str, Any],
        claim: dict[str, Any],
        audit_trail: Optional[dict[str, Any]] = None,
        zkp_proofs: Optional[dict[str, Any]] = None,
    ) -> ReceiptResult:
        """Generate a receipt for a payment.

        Parameters
        ----------
        payment_record : dict
            The payment record from the ledger (includes payment_id,
            amounts, ACH reference, etc.).
        claim : dict
            The original claim data (claimant, policy, date of loss, etc.).
        audit_trail : dict, optional
            The full agent trace (from the audit record assembler).
        zkp_proofs : dict, optional
            References to the ZKP proofs generated during processing.
        """
        receipt_id = f"RCP-{payment_record.get('payment_id', 'XXXX')}"
        timestamp = time.time()

        if _HAS_REPORTLAB:
            return self._generate_pdf(
                receipt_id=receipt_id,
                payment_record=payment_record,
                claim=claim,
                audit_trail=audit_trail or {},
                zkp_proofs=zkp_proofs or {},
                timestamp=timestamp,
            )
        else:
            return self._generate_text(
                receipt_id=receipt_id,
                payment_record=payment_record,
                claim=claim,
                audit_trail=audit_trail or {},
                zkp_proofs=zkp_proofs or {},
                timestamp=timestamp,
            )

    def _generate_pdf(
        self,
        *,
        receipt_id: str,
        payment_record: dict[str, Any],
        claim: dict[str, Any],
        audit_trail: dict[str, Any],
        zkp_proofs: dict[str, Any],
        timestamp: float,
    ) -> ReceiptResult:
        """Generate a PDF receipt using ReportLab."""
        file_path = self.output_dir / f"{receipt_id}.pdf"
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer, pagesize=letter,
                leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                topMargin=0.75 * inch, bottomMargin=0.75 * inch,
            )
            styles = getSampleStyleSheet()
            story = []

            # ---- Title ----
            title_style = ParagraphStyle(
                "ReceiptTitle", parent=styles["Title"],
                fontSize=18, spaceAfter=6, alignment=TA_CENTER,
            )
            story.append(Paragraph("ShieldPoint Insurance", title_style))
            story.append(Paragraph("Claim Payment Receipt", styles["Heading2"]))
            story.append(Spacer(1, 0.2 * inch))

            # ---- Receipt metadata ----
            meta_data = [
                ["Receipt ID:", receipt_id],
                ["Payment ID:", payment_record.get("payment_id", "N/A")],
                ["Date:", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(timestamp))],
                ["Status:", payment_record.get("status", "N/A").upper()],
            ]
            meta_table = Table(meta_data, colWidths=[1.5 * inch, 4 * inch])
            meta_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 0.3 * inch))

            # ---- Claimant information ----
            story.append(Paragraph("Claimant Information", styles["Heading3"]))
            claimant_data = [
                ["Claimant:", claim.get("claimant", "N/A")],
                ["Policy ID:", claim.get("policy_id", "N/A")],
                ["Claim ID:", claim.get("claim_id", "N/A")],
                ["Date of Loss:", claim.get("date_of_loss", "N/A")],
                ["Claim Type:", claim.get("claim_type", "N/A")],
            ]
            claimant_table = Table(claimant_data, colWidths=[1.5 * inch, 4 * inch])
            claimant_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(claimant_table)
            story.append(Spacer(1, 0.3 * inch))

            # ---- Payment breakdown ----
            story.append(Paragraph("Payment Breakdown", styles["Heading3"]))
            breakdown_data = [
                ["Description", "Amount"],
                ["Gross Claim Amount", f"${payment_record.get('gross_amount', 0):,.2f}"],
                ["Deductible Applied", f"-${payment_record.get('deductible_applied', 0):,.2f}"],
                ["Co-pay", f"-${payment_record.get('copay_amount', 0):,.2f}"],
                ["", ""],
                ["Net Payable (ACH)", f"${payment_record.get('net_payable', 0):,.2f}"],
            ]
            breakdown_table = Table(breakdown_data, colWidths=[4 * inch, 1.5 * inch])
            breakdown_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEABOVE", (0, -1), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(breakdown_table)
            story.append(Spacer(1, 0.3 * inch))

            # ---- ACH details ----
            story.append(Paragraph("Payment Details", styles["Heading3"]))
            ach_data = [
                ["ACH Reference:", payment_record.get("ach_reference", "N/A")],
                ["Expected Settlement:", payment_record.get("settlement_date", "N/A")],
                ["Payment Method:", "ACH (Automated Clearing House)"],
            ]
            ach_table = Table(ach_data, colWidths=[1.5 * inch, 4 * inch])
            ach_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ach_table)
            story.append(Spacer(1, 0.3 * inch))

            # ---- ZKP proof references ----
            if zkp_proofs:
                story.append(Paragraph("ZKP Proof References", styles["Heading3"]))
                proof_data = [["Proof Type", "Verified", "Reference"]]
                for proof_type, proof_info in zkp_proofs.items():
                    verified = "Yes" if proof_info.get("verified") else "No"
                    ref = proof_info.get("reference", "N/A")
                    proof_data.append([proof_type, verified, ref])
                proof_table = Table(proof_data, colWidths=[2 * inch, 1 * inch, 2.5 * inch])
                proof_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(proof_table)
                story.append(Spacer(1, 0.3 * inch))

            # ---- Audit trail ----
            if audit_trail:
                story.append(Paragraph("Processing Audit Trail", styles["Heading3"]))
                agent_data = [["Agent", "State", "Timestamp", "Outcome"]]
                for entry in audit_trail.get("agent_traces", []):
                    agent_data.append([
                        entry.get("agent", "N/A"),
                        entry.get("state", "N/A"),
                        entry.get("timestamp", "N/A"),
                        entry.get("outcome", "N/A"),
                    ])
                agent_table = Table(agent_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1 * inch])
                agent_table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                story.append(agent_table)
                story.append(Spacer(1, 0.3 * inch))

            # ---- Footer ----
            footer_style = ParagraphStyle(
                "Footer", parent=styles["Normal"],
                fontSize=8, textColor=colors.grey, alignment=TA_CENTER,
            )
            story.append(Spacer(1, 0.5 * inch))
            story.append(Paragraph(
                "This receipt is generated electronically by the ShieldPoint "
                "Claims Automation System. Please retain for your records.",
                footer_style,
            ))
            story.append(Paragraph(
                f"Receipt ID: {receipt_id} | "
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(timestamp))}",
                footer_style,
            ))

            doc.build(story)
            pdf_bytes = buffer.getvalue()

            with open(file_path, "wb") as f:
                f.write(pdf_bytes)

            return ReceiptResult(
                success=True,
                file_path=str(file_path),
                file_format="pdf",
                receipt_id=receipt_id,
                generated_at=timestamp,
            )

        except Exception as e:
            logger.error("PDF generation failed: %s — falling back to text", e)
            return self._generate_text(
                receipt_id=receipt_id,
                payment_record=payment_record,
                claim=claim,
                audit_trail=audit_trail,
                zkp_proofs=zkp_proofs,
                timestamp=timestamp,
                error=str(e),
            )

    def _generate_text(
        self,
        *,
        receipt_id: str,
        payment_record: dict[str, Any],
        claim: dict[str, Any],
        audit_trail: dict[str, Any],
        zkp_proofs: dict[str, Any],
        timestamp: float,
        error: Optional[str] = None,
    ) -> ReceiptResult:
        """Generate a plain-text receipt (fallback when ReportLab is unavailable)."""
        file_path = self.output_dir / f"{receipt_id}.txt"
        lines = [
            "=" * 60,
            "ShieldPoint Insurance — Claim Payment Receipt",
            "=" * 60,
            "",
            f"Receipt ID:     {receipt_id}",
            f"Payment ID:     {payment_record.get('payment_id', 'N/A')}",
            f"Date:           {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(timestamp))}",
            f"Status:         {payment_record.get('status', 'N/A').upper()}",
            "",
            "-" * 60,
            "Claimant Information",
            "-" * 60,
            f"Claimant:       {claim.get('claimant', 'N/A')}",
            f"Policy ID:      {claim.get('policy_id', 'N/A')}",
            f"Claim ID:       {claim.get('claim_id', 'N/A')}",
            f"Date of Loss:   {claim.get('date_of_loss', 'N/A')}",
            f"Claim Type:     {claim.get('claim_type', 'N/A')}",
            "",
            "-" * 60,
            "Payment Breakdown",
            "-" * 60,
            f"Gross Claim Amount:    ${payment_record.get('gross_amount', 0):,.2f}",
            f"Deductible Applied:    -${payment_record.get('deductible_applied', 0):,.2f}",
            f"Co-pay:                -${payment_record.get('copay_amount', 0):,.2f}",
            f"                            --------",
            f"Net Payable (ACH):     ${payment_record.get('net_payable', 0):,.2f}",
            "",
            "-" * 60,
            "Payment Details",
            "-" * 60,
            f"ACH Reference:    {payment_record.get('ach_reference', 'N/A')}",
            f"Expected Settlement: {payment_record.get('settlement_date', 'N/A')}",
            f"Payment Method:   ACH (Automated Clearing House)",
            "",
        ]

        if zkp_proofs:
            lines.append("-" * 60)
            lines.append("ZKP Proof References")
            lines.append("-" * 60)
            for proof_type, proof_info in zkp_proofs.items():
                verified = "Yes" if proof_info.get("verified") else "No"
                ref = proof_info.get("reference", "N/A")
                lines.append(f"{proof_type}: verified={verified}, ref={ref}")
            lines.append("")

        if audit_trail:
            lines.append("-" * 60)
            lines.append("Processing Audit Trail")
            lines.append("-" * 60)
            for entry in audit_trail.get("agent_traces", []):
                lines.append(
                    f"  {entry.get('agent', 'N/A')} / {entry.get('state', 'N/A')} "
                    f"@ {entry.get('timestamp', 'N/A')} -> {entry.get('outcome', 'N/A')}"
                )
            lines.append("")

        lines.extend([
            "=" * 60,
            "This receipt is generated electronically by the ShieldPoint",
            "Claims Automation System. Please retain for your records.",
            "=" * 60,
        ])

        file_path.write_text("\n".join(lines))

        return ReceiptResult(
            success=True,
            file_path=str(file_path),
            file_format="txt",
            receipt_id=receipt_id,
            generated_at=timestamp,
            error=error,
        )
