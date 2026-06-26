#!/usr/bin/env python3
"""Generate a sample fax PDF for manual testing.

Usage::

    python scripts/make_sample_fax.py > samples/sample_fax.pdf
"""

from __future__ import annotations

import io
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


def build_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "SHIELDPOINT CLAIM INTAKE FORM")
    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(72, y, "Fax received: 2026-03-15 08:42 EST")
    y -= 36

    c.setFont("Helvetica", 11)
    fields = [
        ("Policyholder Name:", "Alice Homeowner"),
        ("Policy ID:", "HO-2024-001"),
        ("Claim Type:", "homeowners"),
        ("Date of Loss:", "2026-03-14"),
        ("Amount Claimed:", "$1,250.00"),
        ("Phone:", "(555) 123-4567"),
        ("Email:", "alice@example.com"),
        ("Location:", "123 Main St, Anytown, USA"),
    ]
    for label, value in fields:
        c.drawString(72, y, f"{label} {value}")
        y -= 18

    y -= 12
    c.drawString(72, y, "Damage Description:")
    y -= 18

    description = (
        "Wind damage to roof shingles during severe thunderstorm on "
        "March 14, 2026. Approximately 30% of shingles blown off the "
        "south-facing roof slope. Water intrusion noted in upstairs "
        "bedroom. Emergency tarp installed by contractor."
    )
    for line in simpleSplit(description, "Helvetica", 11, width - 144):
        c.drawString(72, y, line)
        y -= 14

    c.showPage()
    c.save()
    return buf.getvalue()


if __name__ == "__main__":
    sys.stdout.buffer.write(build_pdf())
