"""Utilities for creating small redistributable PDFs in tests."""

import pymupdf


def make_pdf(pages: list[str]) -> bytes:
    """Create an in-memory PDF with one text block per page."""
    pdf = pymupdf.open()
    try:
        for text in pages:
            page = pdf.new_page()
            page.insert_textbox(
                pymupdf.Rect(36, 36, 559, 806),
                text,
                fontsize=11,
            )
        return pdf.tobytes()
    finally:
        pdf.close()
