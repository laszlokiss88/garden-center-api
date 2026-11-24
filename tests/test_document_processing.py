"""Unit tests for PDF validation, extraction, and chunking."""

import pytest
import tiktoken

from garden_center.document_processing import (
    DocumentValidationError,
    ExtractedPage,
    chunk_pages,
    extract_pdf_pages,
    validate_upload,
)
from tests.pdf_factory import make_pdf


def test_extracts_normalized_text_with_page_numbers() -> None:
    content = make_pdf(["First page text.", "Second page text."])

    pages = extract_pdf_pages(content)

    assert [(page.page_number, page.text) for page in pages] == [
        (1, "First page text."),
        (2, "Second page text."),
    ]


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "expected_code"),
    [
        ("empty.pdf", "application/pdf", b"", "empty_file"),
        ("notes.txt", "application/pdf", b"%PDF-data", "invalid_file_type"),
        ("notes.pdf", "text/plain", b"%PDF-data", "invalid_content_type"),
        ("notes.pdf", "application/pdf", b"not a pdf", "invalid_pdf"),
        ("large.pdf", "application/pdf", b"%PDF-too-large", "file_too_large"),
    ],
)
def test_rejects_invalid_uploads(
    filename: str,
    content_type: str,
    content: bytes,
    expected_code: str,
) -> None:
    with pytest.raises(DocumentValidationError) as error:
        max_size = 8 if expected_code == "file_too_large" else 1024
        validate_upload(filename, content_type, content, max_size)

    assert error.value.code == expected_code


def test_rejects_malformed_pdf() -> None:
    with pytest.raises(DocumentValidationError) as error:
        extract_pdf_pages(b"%PDF-1.7\nmalformed")

    assert error.value.code == "malformed_pdf"


def test_rejects_pdf_without_extractable_text() -> None:
    with pytest.raises(DocumentValidationError) as error:
        extract_pdf_pages(make_pdf([""]))

    assert error.value.code == "no_extractable_text"


def test_chunk_size_overlap_order_and_source_page() -> None:
    pages = [
        ExtractedPage(1, " ".join(f"alpha{i}" for i in range(80))),
        ExtractedPage(2, " ".join(f"beta{i}" for i in range(30))),
    ]
    chunks = chunk_pages(pages, chunk_size=30, overlap=5)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.token_count <= 30 for chunk in chunks)
    assert {chunk.source_page for chunk in chunks} == {1, 2}

    first_page_chunks = [chunk for chunk in chunks if chunk.source_page == 1]
    first_tokens = encoding.encode(first_page_chunks[0].text)
    assert encoding.decode(first_tokens[-4:]).strip() in first_page_chunks[1].text
