from __future__ import annotations

from typing import Any

import fitz


def extract_pdf_pages(pdf_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Extract text from a PDF page by page using PyMuPDF.

    Returns one dictionary per page so later steps can preserve page-level source
    citations in ChromaDB metadata.
    """
    pages: list[dict[str, Any]] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            pages.append(
                {
                    "filename": filename,
                    "page_number": page_index,
                    "text": text,
                }
            )

    return pages
