"""Pure PDF extraction, normalization, validation, and chunking utilities."""

import re
from dataclasses import dataclass

import pymupdf
import tiktoken

PDF_CONTENT_TYPE = "application/pdf"
_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class DocumentValidationError(ValueError):
    """Raised when an uploaded document cannot be safely processed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Normalized text extracted from one source page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A token-bounded piece of text with source metadata."""

    chunk_index: int
    text: str
    source_page: int
    token_count: int


def validate_upload(
    filename: str | None,
    content_type: str | None,
    content: bytes,
    max_size_bytes: int,
) -> None:
    """Validate basic PDF upload metadata and bytes before parsing."""
    if not content:
        raise DocumentValidationError("empty_file", "The uploaded file is empty.")
    if len(content) > max_size_bytes:
        raise DocumentValidationError(
            "file_too_large",
            f"The uploaded file exceeds the {max_size_bytes}-byte limit.",
        )
    if not filename or not filename.lower().endswith(".pdf"):
        raise DocumentValidationError("invalid_file_type", "A .pdf file is required.")
    if content_type != PDF_CONTENT_TYPE:
        raise DocumentValidationError(
            "invalid_content_type", "Content-Type must be application/pdf."
        )
    if not content.startswith(b"%PDF-"):
        raise DocumentValidationError("invalid_pdf", "The file is not a valid PDF.")


def normalize_text(text: str) -> str:
    """Normalize PDF whitespace while preserving paragraph boundaries."""
    normalized_lines = [
        _WHITESPACE.sub(" ", line).strip() for line in text.splitlines()
    ]
    normalized = "\n".join(normalized_lines)
    normalized = _EXCESS_NEWLINES.sub("\n\n", normalized)
    return normalized.strip()


def extract_pdf_pages(content: bytes) -> list[ExtractedPage]:
    """Extract normalized text with one-based page references."""
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            pages = [
                ExtractedPage(page.number + 1, normalize_text(page.get_text("text")))
                for page in pdf
            ]
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise DocumentValidationError(
            "malformed_pdf", "The PDF is malformed or cannot be read."
        ) from exc

    if not pages or not any(page.text for page in pages):
        raise DocumentValidationError(
            "no_extractable_text",
            "The PDF contains no extractable text; OCR is not supported.",
        )
    return pages


def _preferred_break(text: str, minimum_character: int) -> int | None:
    paragraph = text.rfind("\n\n", minimum_character)
    if paragraph >= minimum_character:
        return paragraph + 2
    sentence_ends = [match.end() for match in _SENTENCE_BOUNDARY.finditer(text)]
    candidates = [end for end in sentence_ends if end >= minimum_character]
    return candidates[-1] if candidates else None


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 500,
    overlap: int = 75,
) -> list[TextChunk]:
    """Split pages deterministically into token-aware chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[TextChunk] = []

    for page in pages:
        tokens = encoding.encode(page.text)
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            candidate_tokens = tokens[start:end]
            candidate_text = encoding.decode(candidate_tokens)

            if end < len(tokens):
                break_at = _preferred_break(
                    candidate_text, int(len(candidate_text) * 0.6)
                )
                if break_at is not None:
                    shortened = encoding.encode(candidate_text[:break_at].rstrip())
                    if shortened:
                        candidate_tokens = shortened
                        end = start + len(shortened)

            text = encoding.decode(candidate_tokens).strip()
            if text:
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        text=text,
                        source_page=page.page_number,
                        token_count=len(encoding.encode(text)),
                    )
                )

            if end >= len(tokens):
                break
            next_start = end - overlap
            start = next_start if next_start > start else end

    return chunks
